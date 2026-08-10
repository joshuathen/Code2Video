from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Gradient descent uses the blame to adjust the knobs.",
            "We nudge weights in the opposite direction of error.",
            "Imagine a ball rolling down toward the lowest loss.",
            "A small learning rate ensures we don't overcorrect.",
            "Each tiny adjustment brings us closer to perfection."
        ]
        self.setup_layout("Gradient Descent: Turning the Knobs", lecture_lines)

        # === Persistent Objects Initialization ===
        
        # 1. Loss Plot & Ball
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": False, "color": BLUE_B, "stroke_width": 2},
            tips=False
        )
        parabola = axes.plot(lambda x: x**2, x_range=[-2, 2], color=BLUE_A)
        ball_pos = ValueTracker(1.5)
        ball = Dot(color=RED, radius=0.12)
        ball.add_updater(lambda d: d.move_to(axes.c2p(ball_pos.get_value(), ball_pos.get_value()**2)))
        
        loss_plot = VGroup(axes, parabola, ball)
        self.place_in_area(loss_plot, "B2", "D5", scale_factor=0.9)

        # 2. Dials (Knobs)
        dial_tracker = ValueTracker(0) # Rotation angle in radians
        
        def create_dial(label_text):
            circle = Circle(radius=0.35, color=WHITE, stroke_width=3)
            # Tick mark inside the dial - persistent object
            tick = Line(ORIGIN, [0, 0.35, 0], color=YELLOW, stroke_width=4)
            # Reference dot at center
            center_dot = Dot(radius=0.04, color=GRAY)
            dial_body = VGroup(circle, tick, center_dot)
            
            # The rotation logic
            tick.add_updater(lambda t: t.set_angle(PI/2 - dial_tracker.get_value()))
            
            label = Text(label_text, font_size=18, color=WHITE).next_to(circle, DOWN, buff=0.15)
            return VGroup(dial_body, label)

        dial1 = create_dial("Weight 1")
        dial2 = create_dial("Weight 2")
        dials = VGroup(dial1, dial2).arrange(RIGHT, buff=0.8)
        self.place_in_area(dials, "E2", "E5", scale_factor=1.0)

        # 3. Learning Rate
        lr_text = Text("Learning Rate", font_size=18, color=GREEN_A)
        lr_val = MathTex(r"\eta = 0.1", font_size=22, color=GREEN_A)
        lr_arrow = Arrow(LEFT, RIGHT, color="#00FF00", buff=0, stroke_width=3).scale(0.4)
        lr_group = VGroup(lr_text, lr_val, lr_arrow).arrange(RIGHT, buff=0.2)
        self.place_at_grid(lr_group, "F3", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # "Gradient descent uses the blame to adjust the knobs."
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(dials), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We nudge weights in the opposite direction of error."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # 'Blame' indicator (Gradient arrow)
        blame_label = Text("Blame", font_size=16, color=RED_B)
        blame_arrow = Arrow(UP, DOWN, color=RED_B, buff=0).scale(0.5)
        blame_group = VGroup(blame_label, blame_arrow).arrange(DOWN, buff=0.1)
        self.place_at_grid(blame_group, "D6", scale_factor=1.0)
        
        self.play(FadeIn(blame_group))
        self.play(
            dial_tracker.animate.set_value(0.6), 
            run_time=1.5,
            rate_func=smooth
        )
        self.play(FadeOut(blame_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Imagine a ball rolling down toward the lowest loss."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(Create(axes), Create(parabola), FadeIn(ball))
        self.play(
            ball_pos.animate.set_value(0.6), 
            dial_tracker.animate.set_value(1.2),
            run_time=2, 
            rate_func=bezier([0,0,1,1])
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "A small learning rate ensures we don't overcorrect."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        self.play(FadeIn(lr_group))
        # Small step
        self.play(
            ball_pos.animate.set_value(0.3), 
            dial_tracker.animate.set_value(1.5),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Each tiny adjustment brings us closer to perfection."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Final descent
        self.play(
            ball_pos.animate.set_value(0), 
            dial_tracker.animate.set_value(1.8),
            run_time=2
        )
        self.wait(3)
        self.lecture[4].set_color(WHITE)
