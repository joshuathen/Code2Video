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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout with section title and lecture lines
        self.setup_layout("Linear Independence: The Essential Team", [
            "Linearly independent vectors provide unique, essential directions.",
            "No vector in the set depends on the others.",
            "Adding an independent vector increases the dimension spanned."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show vectors 'A' (East, #FFD700) and 'B' (North, #00BFFF) on a 2D plane.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Define grid-based positions
        origin_pos = self.grid["E2"]
        vec_a_pos = self.grid["E4"]
        vec_b_pos = self.grid["C2"]
        vec_ab_pos = self.grid["C4"]
        
        vector_a = Arrow(origin_pos, vec_a_pos, color="#FFD700", buff=0)
        vector_b = Arrow(origin_pos, vec_b_pos, color="#00BFFF", buff=0)
        
        # Labels within 1 unit of tips
        label_a = MathTex("A", color="#FFD700").scale(0.8)
        self.place_at_grid(label_a, "E5") # 1 unit east of E4
        
        label_b = MathTex("B", color="#00BFFF").scale(0.8)
        self.place_at_grid(label_b, "B2") # 1 unit north of C2
        
        # Create a parallelogram representing the span (plane)
        span_plane = Polygon(
            origin_pos, vec_a_pos, vec_ab_pos, vec_b_pos,
            color=WHITE, stroke_width=1, fill_opacity=0.1
        )
        
        self.play(GrowArrow(vector_a), GrowArrow(vector_b))
        self.play(Write(label_a), Write(label_b))
        self.play(FadeIn(span_plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Introduce vector 'D' in #ADFF2F pointing 'Up' out of the plane.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Simulate 3D "Up" with a diagonal vector
        vec_d_pos = self.grid["D3"]
        vector_d = Arrow(origin_pos, vec_d_pos, color="#ADFF2F", buff=0)
        label_d = MathTex("D", color="#ADFF2F").scale(0.8)
        self.place_at_grid(label_d, "C3") # 1 unit north of D3
        
        self.play(GrowArrow(vector_d))
        self.play(Write(label_d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Expand the 2D plane into a translucent 3D volume (cube) to show the new span.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Define vertices for the volume (parallelepiped)
        v0 = origin_pos
        v1 = vec_a_pos
        v2 = vec_b_pos
        v3 = vec_d_pos
        v12 = vec_ab_pos
        v13 = self.grid["D5"]
        v23 = self.grid["B3"]
        v123 = self.grid["B5"]
        
        # Faces of the 3D span
        face_back = Polygon(v3, v13, v123, v23, color="#ADFF2F", fill_opacity=0.1, stroke_width=1)
        face_bottom_back = Polygon(v0, v1, v13, v3, color="#ADFF2F", fill_opacity=0.1, stroke_width=1)
        face_top = Polygon(v2, v12, v123, v23, color="#ADFF2F", fill_opacity=0.1, stroke_width=1)
        face_left = Polygon(v0, v2, v23, v3, color="#ADFF2F", fill_opacity=0.1, stroke_width=1)
        face_right = Polygon(v1, v12, v123, v13, color="#ADFF2F", fill_opacity=0.1, stroke_width=1)
        
        volume_faces = VGroup(face_back, face_bottom_back, face_top, face_left, face_right)
        
        # Ensure labels and vectors remain visible
        self.add(vector_a, vector_b, vector_d, label_a, label_b, label_d)
        
        self.play(FadeIn(volume_faces))
        self.wait(2)
        
        # Cleanup: Return lecture to white
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
