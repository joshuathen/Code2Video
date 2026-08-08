from manim import *

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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title = "The Geometric Setup of Ax = b"
        lines = [
            "Ax equals b scales our basis vectors.",
            "We stretch v1 by x and v2 by y.",
            "Their sum reaches the target vector b."
        ]
        self.setup_layout(title, lines)
        
        # Define Coordinates and Colors
        # Origin at F1 (math 0,0)
        # v1 (2,1) tip at E3 (x=0.5+2, y=2.2-4)
        # v2 (1,2) tip at D2 (x=0.5+1, y=2.2-3)
        # b (4,5) tip at A5 (x=0.5+4, y=2.2-0)
        # 2*v2 (2,4) tip at B3 (x=0.5+2, y=2.2-1)
        
        origin = self.grid["F1"]
        v1_end = self.grid["E3"]
        v2_end = self.grid["D2"]
        b_end = self.grid["A5"]
        y_v2_scaled_end = self.grid["B3"]
        
        V1_COLOR = "#FFFF00"  # Yellow
        V2_COLOR = "#00FF00"  # Green
        B_COLOR = "#FF00FF"   # Magenta
        TEXT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(V1_COLOR))
        
        # Background NumberPlane centered in the A1-F6 area
        plane = NumberPlane(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'A1', 'F6')
        
        # Initial vectors v1, v2, and target b
        v1 = Arrow(origin, v1_end, buff=0, color=V1_COLOR)
        v2 = Arrow(origin, v2_end, buff=0, color=V2_COLOR)
        b_vec = Arrow(origin, b_end, buff=0, color=B_COLOR)
        
        v1_label = MathTex("v_1", color=V1_COLOR)
        self.place_at_grid(v1_label, "E4", scale_factor=0.6)
        
        v2_label = MathTex("v_2", color=V2_COLOR)
        # Fixed Issue 25: Moved v2_label to D2 per critic suggestion
        self.place_at_grid(v2_label, "D2", scale_factor=0.6)
        
        b_label = MathTex("b", color=B_COLOR)
        self.place_at_grid(b_label, "A6", scale_factor=0.7)
        
        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(v1), GrowArrow(v2))
        self.play(Write(v1_label), Write(v2_label))
        self.wait(0.5)
        self.play(GrowArrow(b_vec))
        self.play(Write(b_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We stretch v1 by x and v2 by y."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(V2_COLOR)
        )
        
        # Target scaled vector y*v2 (2,4)
        y_v2_vec = Arrow(origin, y_v2_scaled_end, buff=0, color=V2_COLOR)
        
        # Scaling animation
        self.play(
            Transform(v2, y_v2_vec),
            v1.animate.set_stroke(width=6),
            v2_label.animate.move_to(self.grid["C2"])
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Their sum reaches the target vector b."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(B_COLOR)
        )
        
        # Tip-to-tail sum
        target_sum_pos = Arrow(v1_end, b_end, buff=0, color=V2_COLOR)
        
        x_v1_label = MathTex("x \\cdot v_1", color=TEXT_COLOR)
        # Fixed Issue 26: Moved x_v1_label to F4 for better alignment
        self.place_at_grid(x_v1_label, "F4", scale_factor=0.6)
        
        y_v2_label = MathTex("y \\cdot v_2", color=TEXT_COLOR)
        self.place_at_grid(y_v2_label, "C5", scale_factor=0.6)
        
        self.play(
            v2.animate.move_to(target_sum_pos.get_center()),
            FadeOut(v1_label),
            FadeOut(v2_label)
        )
        self.play(
            FadeIn(x_v1_label),
            FadeIn(y_v2_label)
        )
        self.wait(2)
