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

class Section5Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout and Teaching Content
        title_text = "Application: The Rabbit Colony Growth"
        lecture_lines = [
            'Consider a growing population of rabbits.',
            'The growth rate depends on the current population.',
            'More rabbits lead to faster reproduction.',
            'This relationship creates a specific growth curve.',
            'We call this model exponential growth.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # 2. Objects Creation
        # Axes Group (Issue 47 fix)
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 100, 20],
            x_length=6,
            y_length=4.5,
            axis_config={"include_tip": True, "color": WHITE},
            tips=True
        ).add_coordinates()
        
        # Labels for the axes
        x_label = Text("t (time)", font_size=18).next_to(axes.x_axis.get_end(), DOWN)
        y_label = Text("P (pop.)", font_size=18).next_to(axes.y_axis.get_end(), LEFT).rotate(90*DEGREES)
        axes_group = VGroup(axes, x_label, y_label)
        # Grid Fix: Issue 47
        self.place_in_area(axes_group, 'B1', 'F6', scale_factor=0.8) 
        
        # Formulas
        # Grid Fix: Issue 45 (dP_dt_formula at A1)
        dP_dt_formula = Text("dP/dt = kP", color="#00FFFF", font_size=36)
        self.place_at_grid(dP_dt_formula, 'A1', scale_factor=0.6)
        
        # Grid Fix: Issue 46 (exponential_formula at A4)
        exponential_formula = Text("P(t) = P0 * e^(kt)", color="#00FF00", font_size=36)
        self.place_at_grid(exponential_formula, 'A4', scale_factor=0.6)

        # Asset path from Issue 38
        rabbit_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/rabbit.svg"
        
        # Initial rabbits (Line 1) - WHITE
        initial_rabbits = VGroup()
        for _ in range(3):
            r = SVGMobject(rabbit_asset, color=WHITE).scale(0.15)
            offset = np.array([np.random.uniform(-0.15, 0.15), np.random.uniform(-0.15, 0.15), 0])
            r.move_to(axes.c2p(0, 10) + offset)
            initial_rabbits.add(r)

        # Growth Curve calculation
        p0 = 10
        k_val = 0.55
        growth_curve = axes.plot(lambda t: p0 * np.exp(k_val * t), x_range=[0, 4.2], color="#00FF00")

        # Duplicating rabbits (Line 3) - PINK (#FFC0CB) from Issue 38
        extra_rabbits = VGroup()
        times = np.linspace(0.5, 4.1, 20)
        for t in times:
            pop = p0 * np.exp(k_val * t)
            # Spawn logic to simulate exponential growth density
            num_to_add = 1 + int(pop / 35)
            for _ in range(num_to_add):
                r = SVGMobject(rabbit_asset, color="#FFC0CB").scale(0.12)
                offset = np.array([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0])
                r.move_to(axes.c2p(t, pop) + offset)
                extra_rabbits.add(r)

        # --- Animation Sequences ---

        # === Animation for Lecture Line 1 ===
        # Color: #FFFFFF
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Create(axes_group))
        self.play(FadeIn(initial_rabbits))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color: #00FFFF
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Write(dP_dt_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color: #FFC0CB
        self.play(self.lecture[2].animate.set_color("#FFC0CB"))
        # Duplication animation using LaggedStart for efficient rendering
        self.play(LaggedStartMap(FadeIn, extra_rabbits, lag_ratio=0.1, run_time=5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Color: #00FF00
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.play(Write(exponential_formula))
        self.play(Create(growth_curve), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Color: #FFFF00
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        # Relationship highlighting effect
        flash_rect = SurroundingRectangle(dP_dt_formula, color="#FFFF00")
        self.play(Create(flash_rect))
        self.play(
            Indicate(growth_curve, color="#FFFF00"), 
            Flash(flash_rect, color="#FFFF00")
        )
        self.play(FadeOut(flash_rect))
        self.wait(3)
