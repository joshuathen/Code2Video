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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initialize Scene
        lecture_lines = [
            'Flattening the curve means reducing the peak.', 
            'Lowering Beta keeps cases within hospital capacity.', 
            'Math helps us manage and survive outbreaks.'
        ]
        self.setup_layout("Conclusion: Flattening the Curve", lecture_lines)
        
        # Define Colors
        RED_CURVE = "#e74c3c"
        CAPACITY_WHITE = "#ffffff"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Setup Axes - Fixed overlap by using B1-F5 area (Issue 53)
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, "B1", "F5")
        
        # Fixed x_label scale and grid positioning (Issue 55)
        x_label = Text("Time", font_size=18, color=WHITE)
        self.place_at_grid(x_label, "F6", scale_factor=0.8)
        
        y_label = Text("Infected", font_size=18, color=WHITE)
        self.place_at_grid(y_label, "A1", scale_factor=1.0)

        # Healthcare Capacity Line
        capacity_line = DashedLine(
            start=axes.c2p(0, 4),
            end=axes.c2p(9, 4),
            color=CAPACITY_WHITE,
            stroke_width=4
        )
        
        # Fixed capacity text placement and scale (Issue 54)
        capacity_text = Text("Healthcare Capacity", font_size=16, color=CAPACITY_WHITE)
        self.place_in_area(capacity_text, "C2", "C4", scale_factor=0.7)
        
        # Asset integration (Issue 38)
        hospital_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/hospital.svg")
        self.place_at_grid(hospital_icon, "C5", scale_factor=0.4)

        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(capacity_line), FadeIn(capacity_text), FadeIn(hospital_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED_CURVE))
        
        # Tall, narrow red curve (High Beta)
        tall_curve = axes.plot(
            lambda x: 8 * np.exp(-1.5 * (x - 3)**2),
            x_range=[0, 7],
            color=RED_CURVE
        )
        
        tall_label = Text("High Beta", font_size=14, color=RED_CURVE)
        self.place_at_grid(tall_label, "B3", scale_factor=1.0)

        self.play(Create(tall_curve), FadeIn(tall_label), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED_CURVE))
        
        # Flatter, wider red curve (Low Beta)
        flat_curve = axes.plot(
            lambda x: 3 * np.exp(-0.3 * (x - 5)**2),
            x_range=[0, 10],
            color=RED_CURVE
        )
        
        flat_label = Text("Low Beta", font_size=14, color=RED_CURVE)
        self.place_at_grid(flat_label, "D4", scale_factor=1.0)

        self.play(Create(flat_curve), FadeIn(flat_label), run_time=2)
        self.wait(2)
