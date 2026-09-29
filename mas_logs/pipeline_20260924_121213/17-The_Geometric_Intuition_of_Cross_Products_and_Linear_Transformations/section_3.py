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
        lecture_lines = [
            "The scalar triple product computes volume.",
            "Cross products relate to volumetric transformations.",
            "It scales the volume of the unit cube.",
            "Linear transformations squash or stretch space volume.",
            "Cross products describe face normals in 3D."
        ]
        self.setup_layout("The Linear Transformation Link: Determinants in 3D", lecture_lines)
        
        # Use SVG asset for cube
        cube_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg"
        cube = SVGMobject(cube_svg)
        self.place_in_area(cube, 'D3', 'F5', scale_factor=1.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cube))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        # Transformation
        matrix = [[1.5, 0.5, 0], [0.5, 1.2, 0], [0, 0, 1]]
        # Apply transformation to the SVG cube
        cube.apply_matrix(matrix)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        volume_text = Text("Volume", font_size=24, color="#FFFF00")
        self.place_at_grid(volume_text, 'C2')
        self.add(volume_text)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        # Highlight face (simulated with a colored rectangle)
        face_highlight = Rectangle(width=1.0, height=1.0, color="#FF00FF", fill_opacity=0.5)
        self.place_at_grid(face_highlight, 'D4')
        self.play(Create(face_highlight))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.play(FadeOut(face_highlight), FadeOut(volume_text))
        self.wait(1)
