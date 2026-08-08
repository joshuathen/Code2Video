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

class Section2Scene(TeachingScene):
    def construct(self):
        # 1. Setup
        title_text = "Prerequisite: The State Vector"
        lecture_lines = [
            "We define quantum states using a state vector.",
            "State |0⟩ points right, state |1⟩ points up.",
            "Diagonal vectors represent a blend of both states."
        ]
        self.setup_layout(title_text, lecture_lines)

        # 2. Extract coordinates from grid system
        # Moved origin to D4 to avoid crowding lecture text (Issue 27)
        origin_pos = self.grid["D4"]
        x_end_pos = self.grid["D6"]
        y_end_pos = self.grid["B4"]
        radius = np.linalg.norm(x_end_pos - origin_pos)

        # === Animation for Lecture Line 1 ===
        # Color: BLUE_B (Matching axes)
        self.play(self.lecture[0].animate.set_color(BLUE_B))
        
        # Axis lines constructed using grid points
        # Using a slightly longer end point for the axis tips
        x_axis_line = Arrow(start=origin_pos, end=x_end_pos + RIGHT*0.4, buff=0, color=BLUE_D)
        y_axis_line = Arrow(start=origin_pos, end=y_end_pos + UP*0.4, buff=0, color=BLUE_D)
        
        label_0 = MathTex("|0\\rangle", color=WHITE)
        label_1 = MathTex("|1\\rangle", color=WHITE)
        
        # Grid-based label positioning shifted for D4 origin (Issues 28, 29)
        self.place_at_grid(label_0, "E6", scale_factor=0.8)
        self.place_at_grid(label_1, "B3", scale_factor=0.8)
        
        self.play(
            Create(x_axis_line), 
            Create(y_axis_line), 
            Write(label_0), 
            Write(label_1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color: #00FF00 (Matching vector)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )
        
        # Vector along X-axis
        # Thicker stroke to stand out against background axes
        state_vector = Arrow(start=origin_pos, end=x_end_pos, buff=0, color="#00FF00", stroke_width=8)
        
        self.play(GrowArrow(state_vector))
        self.wait(1)
        
        # Rotate to point along Y-axis (|1> state)
        self.play(Rotate(state_vector, angle=PI/2, about_point=origin_pos))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color: #00FFFF (Matching diagonal vector)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Unit circle on which the state vector rotates
        unit_circle = Circle(radius=radius, color=WHITE, stroke_opacity=0.3)
        self.place_at_grid(unit_circle, "D4") # Origin is now D4 (Issue 27)
        
        self.play(Create(unit_circle))
        
        # Rotate back to 45 degree diagonal position
        # Representing a superposition / blend
        self.play(
            Rotate(state_vector, angle=-PI/4, about_point=origin_pos),
            state_vector.animate.set_color("#00FFFF")
        )
        self.wait(3)
