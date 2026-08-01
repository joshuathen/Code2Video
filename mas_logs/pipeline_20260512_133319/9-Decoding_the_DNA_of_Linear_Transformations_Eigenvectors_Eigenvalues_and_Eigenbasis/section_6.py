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
        # Initial layout setup
        self.setup_layout(
            "Real-world Application & Summary", 
            [
                'Eigenvectors reveal the core DNA of transformations.', 
                "They power Google's search and facial recognition systems.", 
                'Understanding them unlocks the heart of linear algebra.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#52C41A"))
        
        # Flow chart elements: Matrix -> Eigenvectors (#52C41A) -> Eigenvalues (#F5222D)
        matrix_txt = Text("Matrix", font_size=24)
        arrow1 = Text("→", font_size=24)
        eigenvec_txt = Text("Eigenvectors", font_size=24, color="#52C41A")
        arrow2 = Text("→", font_size=24)
        eigenval_txt = Text("Eigenvalues", font_size=24, color="#F5222D")
        
        flow_group = VGroup(matrix_txt, arrow1, eigenvec_txt, arrow2, eigenval_txt).arrange(RIGHT, buff=0.3)
        # Resolved Issue 38: repositioned to avoid obstructing lecture notes
        self.place_in_area(flow_group, 'A1', 'C6', scale_factor=0.7)
        
        self.play(FadeIn(flow_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FAAD14"))
        
        # Facial outline using the provided SVG asset (Issue 24)
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/face.svg
        face = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/face.svg", color="#FAAD14")
        
        # Eigenface Arrows indicating primary directions
        arrow_v = Arrow(start=ORIGIN, end=UP * 1.0, color="#52C41A", buff=0)
        arrow_h = Arrow(start=ORIGIN, end=RIGHT * 0.8, color="#52C41A", buff=0)
        eigen_arrows = VGroup(arrow_v, arrow_h).move_to(face.get_center())
        
        face_viz = VGroup(face, eigen_arrows)
        # Resolved Issue 39: adjusted placement and scale to avoid clipping
        self.place_in_area(face_viz, 'D2', 'F5', scale_factor=0.6)
        
        self.play(Create(face))
        self.play(GrowArrow(arrow_v), GrowArrow(arrow_h))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Final title centered in the right-side grid to avoid obstructing lecture notes
        final_title = Text("The DNA of Linear Transformations", font_size=36, color="#FFFFFF")
        self.place_in_area(final_title, "A1", "F6", scale_factor=0.8)
        
        # Background box for clarity
        final_bg = BackgroundRectangle(final_title, color=BLACK, fill_opacity=0.9, buff=0.3)
        
        self.play(FadeIn(final_bg), FadeIn(final_title))
        self.play(Flash(final_title, color=WHITE, line_length=0.3, flash_radius=2.5))
        self.wait(3)
