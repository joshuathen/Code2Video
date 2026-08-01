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
        title = "The Mathematics of Forward Propagation"
        lines = [
            "Data flows forward through a series of matrix multiplications.",
            "Input vectors are transformed by matrices of weights.",
            "This creates a giant, complex mathematical function.",
            "Every connection contributes to the final result.",
            "Mathematical precision enables the network to 'think'."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Input vector X
        input_vector = MathTex(
            "X = \\begin{bmatrix} x_1 \\\\ x_2 \\\\ x_3 \\end{bmatrix}",
            color="#FFFFFF"
        )
        self.place_at_grid(input_vector, "C2", scale_factor=0.8)
        
        # Weight Matrix W
        weight_matrix = MathTex(
            "W = \\begin{bmatrix} w_{11} & w_{12} & w_{13} \\\\ w_{21} & w_{22} & w_{23} \\\\ w_{31} & w_{32} & w_{33} \\end{bmatrix}",
            color="#FFFFFF"
        )
        self.place_at_grid(weight_matrix, "C4", scale_factor=0.8)
        
        self.play(FadeIn(input_vector), FadeIn(weight_matrix))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Result vector Y = WX
        result_vector = MathTex(
            "Y = \\begin{bmatrix} y_1 \\\\ y_2 \\\\ y_3 \\end{bmatrix}",
            color="#FFFFFF"
        )
        self.place_at_grid(result_vector, "C6", scale_factor=0.8)
        
        # Highlights using simple rectangles over the whole matrix/vector area
        row_highlight = Rectangle(width=1.2, height=0.4, color=BLUE, stroke_width=2).move_to(weight_matrix.get_center() + UP * 0.4)
        col_highlight = Rectangle(width=0.6, height=1.2, color=BLUE, stroke_width=2).move_to(input_vector.get_center() + RIGHT * 0.3)
        
        self.play(Create(row_highlight), Create(col_highlight))
        self.wait(0.5)
        self.play(FadeIn(result_vector))
        self.play(FadeOut(row_highlight), FadeOut(col_highlight))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Group everything and move to a grid position to represent "zooming out"
        current_eq = VGroup(input_vector, weight_matrix, result_vector)
        
        # Multiple matrices cascading
        matrices = VGroup()
        for i in range(3):
            m = MathTex(f"W_{i+1}", color="#FFFFFF")
            self.place_at_grid(m, f"B{i+2}", scale_factor=0.6)
            m.shift(RIGHT * i * 0.3 + DOWN * i * 0.3)
            matrices.add(m)
            
        target_pos = self.grid["A6"]
        
        self.play(
            current_eq.animate.scale(0.4).move_to(target_pos).set_opacity(0.3),
            FadeIn(matrices)
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Show all connections glowing
        connections = VGroup()
        for i in range(len(matrices) - 1):
            line = Line(matrices[i].get_right(), matrices[i+1].get_left(), color="#ADD8E6", stroke_width=2)
            connections.add(line)
            
        glow_effects = VGroup(*[
            line.copy().set_stroke(width=8, opacity=0.3) for line in connections
        ])
        
        self.play(Create(connections), FadeIn(glow_effects))
        self.wait(1)
        self.play(Indicate(connections, color="#ADD8E6", scale_factor=1.1))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Final output flows to a 'Decision' node [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/decision.svg]
        decision_node = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/decision.svg")
        decision_node.set_color("#FFD700")
        self.place_at_grid(decision_node, "E5", scale_factor=0.6)
        
        decision_label = Text("Decision", font_size=16, color="#FFD700")
        decision_label.next_to(decision_node, DOWN, buff=0.2)
        
        # Start the arrow from the last matrix
        flow_line = Arrow(matrices[-1].get_right(), decision_node.get_left(), color="#FFD700", buff=0.1)
        
        self.play(
            FadeIn(decision_node),
            Write(decision_label),
            GrowArrow(flow_line)
        )
        self.play(decision_node.animate.set_fill(opacity=0.8), run_time=0.5)
        self.play(decision_node.animate.set_fill(opacity=0.2), run_time=0.5)
        
        self.wait(3)
        
        # Final state color reset
        self.lecture[4].set_color(WHITE)
        self.wait(2)
