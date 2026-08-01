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
        # Initial Setup
        title_str = "Real-World Power: Why It Matters"
        lines = [
            "CLT lets us predict outcomes across diverse fields.",
            "Factories monitor quality without checking every single item.",
            "Statistical certainty arises from individual uncertainty."
        ]
        self.setup_layout(title_str, lines)

        # Colors for matching
        COLOR_L1 = "#87CEEB" # Sky Blue
        COLOR_L2 = "#90EE90" # Light Green
        COLOR_L3 = "#9370DB" # Medium Purple

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_L1))

        # Conveyor belt - Moved to Row A (Issue 56)
        belt_line = Line(self.grid["A1"], self.grid["A6"], color=GREY_B)
        belt_base = Rectangle(height=0.1, width=5.0, color=GREY_D, fill_opacity=0.5).move_to(belt_line.get_center())
        
        # Label: Process Variance - Moved to B3 (Issue 54)
        variance_label = Text("Process Variance", font_size=20, color=COLOR_L1)
        self.place_at_grid(variance_label, "B3", scale_factor=0.8)

        # Soda Cans - Moved to Row A
        def create_can():
            can_body = RoundedRectangle(corner_radius=0.05, height=0.4, width=0.25, fill_opacity=1, color=COLOR_L1)
            can_tab = Rectangle(height=0.05, width=0.1, color=LIGHT_GREY, fill_opacity=1).next_to(can_body, UP, buff=0)
            return VGroup(can_body, can_tab).scale(0.8)

        cans = VGroup(*[create_can() for _ in range(6)])
        for i, can in enumerate(cans):
            can.move_to(self.grid["A1"] + RIGHT * i * 0.8)

        self.play(Create(belt_base), Create(belt_line), FadeIn(variance_label))
        self.play(FadeIn(cans, shift=RIGHT))

        # Can movement animation
        can_tracker = ValueTracker(0)
        def update_cans(obj):
            t = can_tracker.get_value()
            for i, can in enumerate(obj):
                pos_x = (self.grid["A1"][0] + (i * 0.8 + t)) 
                # Loop cans back to start of belt
                start_x = self.grid["A1"][0]
                end_x = self.grid["A6"][0]
                wrapped_x = start_x + (pos_x - start_x) % (end_x - start_x + 0.8)
                can.set_x(wrapped_x)

        cans.add_updater(update_cans)
        self.play(can_tracker.animate.set_value(2), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_L2))

        # Axes for sampling distribution - Moved to E1-F6 (Issue 55)
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 0.5, 0.1],
            x_length=4,
            y_length=2,
            axis_config={"include_tip": False, "font_size": 18},
            tips=False
        )
        self.place_in_area(ax, "E1", "F6", scale_factor=0.7)
        
        # Sampling Distribution Title - Moved to avoid crowding
        ax_label = Text("Sampling Distribution of Means", font_size=16, color=COLOR_L2)
        self.place_at_grid(ax_label, "D3", scale_factor=1.0)

        # Bell Curve
        curve = ax.plot(lambda x: (1 / (1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / 1)**2), color=COLOR_L2)

        # Sampling batches visual effect - Highlight at Row A where cans are
        batch_highlight = Circle(radius=0.4, color=YELLOW, stroke_width=4).move_to(self.grid["A3"])
        
        self.play(Create(ax), FadeIn(ax_label))
        
        # Simulate batch sampling
        for _ in range(2):
            self.play(FadeIn(batch_highlight, scale=0.5))
            dot = Dot(color=COLOR_L2, radius=0.05).move_to(batch_highlight.get_center())
            self.play(dot.animate.move_to(ax.c2p(np.random.normal(0, 0.5), 0)), FadeOut(batch_highlight))
            self.remove(dot)

        self.play(Create(curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_L3))

        # Highlight 95% area
        area_95 = ax.get_area(curve, x_range=[-1.96, 1.96], color=COLOR_L3, opacity=0.4)
        
        label_95 = Text("95%", font_size=24, color=WHITE).move_to(ax.c2p(0, 0.15))
        certainty_text = Text("Certainty", font_size=18, color=COLOR_L3).next_to(label_95, DOWN, buff=0.1)

        self.play(FadeIn(area_95), Write(label_95), FadeIn(certainty_text))
        
        self.wait(3)

        # Cleanup for end of section
        cans.remove_updater(update_cans)
