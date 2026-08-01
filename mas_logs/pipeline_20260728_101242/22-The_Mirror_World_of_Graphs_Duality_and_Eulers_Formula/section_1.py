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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisites: The Planar Landscape", [
            "A planar graph's edges never cross on a plane.",
            "It has vertices, edges, and bounded or unbounded faces.",
            "This house graph has vertices, edges, and faces.",
            "It has five vertices, six edges, and three faces.",
            "Euler's formula: V minus E plus F equals two."
        ])

        # Colors from Animation Description
        v_color = "#ADD8E6" # light blue
        e_color = WHITE     # white
        f_color = "#90EE90" # soft green

        # === Animation for Lecture Line 1 ===
        # Draw a 'House' graph with 5 light blue vertices and 6 edges.
        self.lecture[0].set_color(YELLOW)
        
        v1 = Dot(color=v_color) # Bottom-left
        v2 = Dot(color=v_color) # Bottom-right
        v3 = Dot(color=v_color) # Top-left square
        v4 = Dot(color=v_color) # Top-right square
        v5 = Dot(color=v_color) # Peak

        # Fixed positions from Issue 30 to avoid crowding
        self.place_at_grid(v1, "D3")
        self.place_at_grid(v2, "D5")
        self.place_at_grid(v3, "C3")
        self.place_at_grid(v4, "C5")
        self.place_at_grid(v5, "B4")

        e1 = Line(v1.get_center(), v2.get_center(), color=e_color)
        e2 = Line(v1.get_center(), v3.get_center(), color=e_color)
        e3 = Line(v2.get_center(), v4.get_center(), color=e_color)
        e4 = Line(v3.get_center(), v4.get_center(), color=e_color)
        e5 = Line(v3.get_center(), v5.get_center(), color=e_color)
        e6 = Line(v4.get_center(), v5.get_center(), color=e_color)

        edges = VGroup(e1, e2, e3, e4, e5, e6)
        vertices = VGroup(v1, v2, v3, v4, v5)
        
        self.play(Create(edges), Create(vertices), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fill the 3 regions (faces) with a soft green glow.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Interior faces
        face1 = Polygon(v1.get_center(), v2.get_center(), v4.get_center(), v3.get_center(), 
                        fill_color=f_color, fill_opacity=0.3, stroke_width=0)
        face2 = Polygon(v3.get_center(), v4.get_center(), v5.get_center(), 
                        fill_color=f_color, fill_opacity=0.3, stroke_width=0)
        
        # Indicator for the infinite outer face
        # Fixed position and size from Issue 31
        inf_face_box = Rectangle(width=4.0, height=2.0, color=f_color, fill_opacity=0.05, stroke_width=2)
        self.place_in_area(inf_face_box, "B2", "D6")

        self.play(FadeIn(face1, face2), run_time=1)
        self.play(Create(inf_face_box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Label each vertex (V), edge (E), and face (F) with white text.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        v_label = Text("V", font_size=24, color=WHITE)
        e_label = Text("E", font_size=24, color=WHITE)
        f_label = Text("F", font_size=24, color=WHITE)

        # Position labels within 1 grid unit of corresponding objects - Fixed positions from Issue 30
        self.place_at_grid(v_label, "D2") # Near v1 (D3)
        self.place_at_grid(e_label, "D4") # Near edge e1 (D3-D5)
        self.place_at_grid(f_label, "C4") # Inside face1 (C3-D5)

        self.play(Write(v_label), Write(e_label), Write(f_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Scale up the text 'V - E + F = 2' while substituting counts 5, 6, 3.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Counts
        v_count = Text("V = 5", font_size=28, color=v_color)
        e_count = Text("E = 6", font_size=28, color=WHITE)
        f_count = Text("F = 3", font_size=28, color=f_color)

        # Fixed positions from Issue 32
        self.place_at_grid(v_count, "E3")
        self.place_at_grid(e_count, "E4")
        self.place_at_grid(f_count, "E5")

        self.play(
            ReplacementTransform(v_label.copy(), v_count),
            ReplacementTransform(e_label.copy(), e_count),
            ReplacementTransform(f_label.copy(), f_count)
        )
        self.wait(0.5)

        # Formula and Substitution
        formula_base = MathTex("V - E + F = 2", font_size=40, color=WHITE)
        self.place_in_area(formula_base, "F3", "F6") # Fixed position from Issue 32
        
        formula_nums = MathTex("5 - 6 + 3 = 2", font_size=40, color=WHITE)
        self.place_in_area(formula_nums, "F3", "F6") # Fixed position from Issue 32

        self.play(Write(formula_base))
        self.play(formula_base.animate.scale(1.2))
        self.play(Transform(formula_base, formula_nums))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Euler's formula: V minus E plus F equals two.
        # Flash the 'infinite' outer face to emphasize it counts as a face.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        formula_final = MathTex("V - E + F = 2", font_size=48, color=WHITE)
        self.place_in_area(formula_final, "F3", "F6")

        self.play(Transform(formula_base, formula_final))
        
        # Final flash of the infinite face
        self.play(Flash(inf_face_box, color=f_color, flash_radius=1.8, line_length=0.5))
        self.wait(2)
