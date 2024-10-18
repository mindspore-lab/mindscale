
"""Learning rate decay and weight decay incr functions."""

import math
import mindspore as ops
import mindspore.common.dtype as mstype

lr_decay_style_list = ["constant", "WSD", "linear", "cosine", "inverse-square-root"]
wd_incr_style_list = ["constant", "linear", "cosine"]


class OptimizerParamScheduler():
    """Anneals learning rate and weight decay"""

    def __init__(self, optimizer, init_lr, max_lr, min_lr,
                 lr_warmup_steps, lr_decay_steps, lr_decay_style,
                 start_wd, end_wd, wd_incr_steps, wd_incr_style,
                 use_checkpoint_opt_param_scheduler=True,
                 override_opt_param_scheduler=False,
                 wsd_decay_steps=None,
                 lr_wsd_decay_style=None):

        # Class values.
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        if self.min_lr < 0.0:
            raise ValueError('min_lr need to be larger than or equal to 0.0, '
                             f'but got {self.min_lr}')
        if self.max_lr < self.min_lr:
            raise ValueError('max_lr need to be larger than or equal to min_lr'
                             f'but got max_lr {self.max_lr} and min_lr {self.min_lr}')
        if self.init_lr > self.max_lr:
            raise ValueError('init_lr need to be smaller than or equal to max_lr '
                             f'but got init_lr {self.init_lr} and max_lr {self.max_lr}')

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style
        if self.lr_decay_steps <= 0:
            raise ValueError('start_wd need to be larger than 0, '
                             f'but got {self.start_wd}')
        if self.lr_warmup_steps >= self.lr_decay_steps:
            raise ValueError('lr_warmup_steps need to be smaller than lr_decay_steps'
                             f'but got lr_warmup_steps {self.lr_warmup_steps} '
                             f'and lr_decay_steps {self.lr_decay_steps}')

        self.lr_decay_style = lr_decay_style
        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        self.start_wd = start_wd
        self.end_wd = end_wd
        if self.start_wd < 0.0:
            raise ValueError('start_wd need to be larger than or equal to 0.0, '
                             f'but got {self.start_wd}')
        if self.end_wd < self.start_wd:
            raise ValueError('end_wd need to be larger than or equal to start_wd'
                             f'but got end_wd {self.end_wd} and start_wd {self.start_wd}')
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            if self.use_checkpoint_opt_param_scheduler:
                raise ValueError('both override and use-checkpoint are set.')

        # Set the learning rate
        self.step(0)
        print('> learning rate decay style: {}'.format(self.lr_decay_style))


    def get_wd(self):
        """ Weight decay incr functions"""
        if self.num_steps > self.wd_incr_steps:
            return self.end_wd

        if self.wd_incr_style == 'constant':
            if self.start_wd != self.end_wd:
                raise ValueError("when wd_incr_style is constant, start_wd need to be equal to end_wd.")
            return self.end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        if incr_ratio < 0.0 or incr_ratio > 1.0:
            raise ValueError(f"incr_ratio should be in range [0.0, 1.0], but got {incr_ratio}")
        delta_wd = self.end_wd - self.start_wd

        if self.wd_incr_style == 'linear':
            coeff = incr_ratio
        elif self.wd_incr_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception('{} weight decay increment style is not supported.'.format(
                self.wd_incr_style))

        return self.start_wd + coeff * delta_wd


    def get_lr(self, param_group):
        """
        Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        """

        max_lr = param_group.get('max_lr', self.max_lr)
        min_lr = param_group.get('min_lr', self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return (
                self.init_lr
                + (
                    (max_lr - self.init_lr)
                    * float(self.num_steps)
                    / float(self.lr_warmup_steps)
                )
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == 'constant':
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == 'inverse-square-root':
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps ** 0.5 / (num_steps ** 0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        if decay_ratio < 0.0 or decay_ratio > 1.0:
            raise ValueError(f"decay_ratio should be in range [0.0, 1.0], but got {decay_ratio}")
        delta_lr = max_lr - min_lr

        if self.lr_decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.lr_decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.lr_decay_style == 'WSD':
            wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
            if self.num_steps <= wsd_anneal_start_:
                coeff = 1.0
            else:
                wsd_steps = self.num_steps - wsd_anneal_start_
                wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
                if self.lr_wsd_decay_style == "linear":
                    coeff = (1.0 - wsd_decay_ratio)
                elif self.lr_wsd_decay_style == "cosine":
                    coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
                elif self.lr_wsd_decay_style == "exponential":
                    coeff = ((2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.lr_decay_style))

        return min_lr + coeff * delta_lr


    def step(self, increment):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_wd = self.get_wd()
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self.get_lr(param_group)
            param_group['lr'].assign_value(ops.Tensor(new_lr * param_group.get('lr_mult', 1.0), dtype=mstype.float32))
            param_group['weight_decay'] = new_wd * param_group.get('wd_mult', 1.0)
            self.optimizer.lrs[group_idx] = param_group['lr']


    def state_dict(self):
        """dict of lr param"""
        state_dict = {
            'max_lr': self.max_lr,
            'lr_warmup_steps': self.lr_warmup_steps,
            'num_steps': self.num_steps,
            'lr_decay_style': lr_decay_style_list.index(self.lr_decay_style),
            'lr_decay_steps': self.lr_decay_steps,
            'min_lr': self.min_lr,
            'start_wd': self.start_wd,
            'end_wd': self.end_wd,
            'wd_incr_style': wd_incr_style_list.index(self.wd_incr_style),
            'wd_incr_steps': self.wd_incr_steps
        }
        return state_dict


    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            print(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            if cls_value != sd_value:
                raise ValueError(f'OptimizerParamScheduler: class input value {cls_value} '
                                 f'and checkpoint value {sd_value} for {name} do not match')
        print(' > using checkpoint value {} for {}'.format(sd_value, name))
        return sd_value

    def load_state_dict(self, sd):
        """load param of lr"""
        new_sd = {}
        for k, v in list(sd.items()):
            if k in ['max_lr', 'lr_warmup_steps', 'num_steps', 'lr_decay_style', 'lr_decay_steps',
                     'min_lr', 'start_wd', 'end_wd', 'wd_incr_style', 'wd_incr_steps']:
                if k in ['lr_warmup_steps', 'num_steps', 'lr_decay_style', 'lr_decay_steps',
                         'wd_incr_style', 'wd_incr_steps']:
                    new_sd[k] = int(v.item())
                else:
                    new_sd[k] = v.item()
                sd.pop(k)

        max_lr_ = new_sd.get('max_lr')
        self.max_lr = self._check_and_set(self.max_lr, max_lr_,
                                          'learning rate')

        self.min_lr = self._check_and_set(self.min_lr, new_sd['min_lr'],
                                          'minimum learning rate')

        lr_warmup_steps_ = new_sd.get('lr_warmup_steps')
        self.lr_warmup_steps = self._check_and_set(
            self.lr_warmup_steps,
            lr_warmup_steps_,
            'warmup iterations'
            )

        lr_decay_steps_ = new_sd.get('lr_decay_steps')
        self.lr_decay_steps = self._check_and_set(
            self.lr_decay_steps, lr_decay_steps_,
            'total number of iterations'
            )

        lr_decay_style_ = new_sd.get('lr_decay_style')

        self.lr_decay_style = self._check_and_set(
            self.lr_decay_style,
            lr_decay_style_list[lr_decay_style_],
            'learning rate decay style'
            )

        num_steps = new_sd.get('num_steps')
        self.step(increment=num_steps)

        if 'start_wd' in new_sd:
            self.start_wd = self._check_and_set(self.start_wd,
                                                new_sd.get('start_wd'),
                                                "start weight decay")
            self.end_wd = self._check_and_set(self.end_wd,
                                              new_sd.get('end_wd'),
                                              "end weight decay")
            self.wd_incr_steps = self._check_and_set(self.wd_incr_steps,
                                                     new_sd.get('wd_incr_steps'),
                                                     "total number of weight decay iterations")
            self.wd_incr_style = self._check_and_set(self.wd_incr_style,
                                                     wd_incr_style_list[new_sd.get('wd_incr_style')],
                                                     "weight decay incr style")
