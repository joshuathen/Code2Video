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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        self.setup_layout("Euler's Formula: Growth Meets Rotation", [
            "What happens when growth moves in an imaginary direction?",
            "Multiplying growth by i turns our movement sideways.",
            "Instead of expanding outward, we orbit in a circle.",
            "This relationship is expressed by Euler's famous formula.",
            "Growth becomes rotation on the complex unit circle."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Replaced MathTex with VGroup of Text to avoid 'latex' dependency
        e_box = VGroup(Text("e"), Text("^"), Text("□"), Text(" ")).arrange(RIGHT, buff=0.05)
        e_i = VGroup(Text("e"), Text("^"), Text("i"), Text(" ")).arrange(RIGHT, buff=0.05)
        e_i[2].set_color("#FF69B4")
        
        # Position using grid
        self.place_at_grid(e_box, "B2", scale_factor=1.5)
        self.place_at_grid(e_i, "B2", scale_factor=1.5)
        
        self.play(Write(e_box))
        self.wait(0.5)
        self.play(Transform(e_box, e_i))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF69B4")
        
        # Create circle components
        circle = Circle(radius=1.2, color="#555555")
        self.place_in_area(circle, "C2", "E4")
        center_point = circle.get_center()
        
        # Vector at (1,0)
        vector = Arrow(start=center_point, end=center_point + RIGHT * 1.2, buff=0, color=WHITE)
        
        # Force 'i' at the tip
        force_arrow = Arrow(
            start=center_point + RIGHT * 1.2, 
            end=center_point + RIGHT * 1.2 + UP * 0.8, 
            color="#FF69B4",
            buff=0
        )
        force_label = Text("i", color="#FF69B4").scale(0.8)
        force_label.move_to(force_arrow.get_end() + RIGHT * 0.3)

        self.play(Create(circle))
        self.play(GrowArrow(vector))
        self.play(GrowArrow(force_arrow), Write(force_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        
        # Define the rotation path
        arc = Arc(radius=1.2, start_angle=0, angle=PI/2, color="#00FFFF", arc_center=center_point)
        
        # Vector tip traces blue arc
        self.play(
            Rotate(vector, angle=PI/2, about_point=center_point),
            Create(arc),
            FadeOut(force_arrow),
            FadeOut(force_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        # Replaced MathTex with VGroup of Text to maintain indexing for coloring
        euler_formula = VGroup(
            Text("e^"), Text("i"), Text("x"), Text(" = "), 
            Text("cos(x)"), Text(" + "), Text("i"), Text("sin(x)")
        ).arrange(RIGHT, buff=0.1)
        
        euler_formula[1].set_color("#FF69B4") # i
        euler_formula[4].set_color("#00FFFF") # cos(x)
        euler_formula[6].set_color("#FF69B4") # i
        euler_formula[7].set_color("#00FFFF") # sin(x)
        
        self.place_in_area(euler_formula, "A2", "A5", scale_factor=0.9)
        self.play(Write(euler_formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        self.play(Indicate(euler_formula))
        self.wait(2)
