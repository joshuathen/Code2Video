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
        self.setup_layout("Unfolding the Impossible", [
            "Higher-dimensional objects can be unfolded like boxes.",
            "A 3D cube flattens into a 2D net.",
            "A 4D tesseract unfolds into an 8-cube structure."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Visual: SVG box for 2D net
        unfolded_cube_box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg", color="#FFFFFF")
        label1 = Text("Unfolded Cube", font_size=18).next_to(unfolded_cube_box, UP)
        group1 = VGroup(unfolded_cube_box, label1)
        self.place_at_grid(group1, 'B2', scale_factor=0.6)
        self.play(Create(unfolded_cube_box), Write(label1))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        cube_3d_square = Square(side_length=1.0, fill_opacity=0.3, color="#00FF00")
        label2 = Text("3D Cube", font_size=18).next_to(cube_3d_square, UP)
        group2 = VGroup(cube_3d_square, label2)
        self.place_at_grid(group2, 'B5', scale_factor=0.6)
        self.play(FadeIn(cube_3d_square), Write(label2))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        # Visual: Faces using the asset
        faces_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg", color="#FFFF00")
        label3 = Text("Cube Faces", font_size=18).next_to(faces_icon, DOWN)
        group3 = VGroup(faces_icon, label3)
        self.place_in_area(group3, 'D2', 'F4', scale_factor=0.7)
        self.play(FadeIn(faces_icon), Write(label3))
        
        self.wait(2)
