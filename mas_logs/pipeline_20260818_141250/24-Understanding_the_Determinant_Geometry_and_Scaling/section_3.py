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
        self.setup_layout("The Sign Significance: Orientation", [
            "A negative determinant means reversed orientation.",
            "Imagine the coordinate system flipped like a mirror.",
            "It's like looking at a reflection."
        ])
        
        # === Animation for Lecture Line 1 ===
        # A negative determinant means reversed orientation.
        self.play(self.lecture[0].animate.set_color("#3498DB"))
        
        # Setup vectors i and j
        i_vec = Vector(RIGHT, color="#3498DB")
        j_vec = Vector(UP, color="#3498DB")
        self.place_in_area(i_vec, "C2", "D3", scale_factor=0.8)
        self.place_in_area(j_vec, "C2", "D3", scale_factor=0.8)
        self.play(Create(i_vec), Create(j_vec))

        # === Animation for Lecture Line 2 ===
        # Imagine the coordinate system flipped like a mirror.
        self.play(self.lecture[1].animate.set_color("#E74C3C"))
        
        # Transform vectors and load asset
        i_vec_flipped = Vector(LEFT, color="#E74C3C")
        self.place_in_area(i_vec_flipped, "C2", "D3", scale_factor=0.8)
        
        mirror_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg", color="#E74C3C")
        self.place_in_area(mirror_icon, "C4", "D5", scale_factor=0.5)
        
        label = Text("Reversed", font_size=20, color="#E74C3C")
        self.place_in_area(label, "B5", "B6", scale_factor=0.8)
        
        self.play(Transform(i_vec, i_vec_flipped), FadeIn(mirror_icon), Write(label))
        
        # === Animation for Lecture Line 3 ===
        # It's like looking at a reflection.
        self.play(self.lecture[2].animate.set_color("#F1C40F"))
        
        # Flash indicator
        rect = Rectangle(width=1, height=1, color="#F1C40F")
        self.place_in_area(rect, "E2", "F5", scale_factor=0.6)
        self.play(Flash(rect.get_center(), color="#F1C40F", line_length=0.2))
        self.wait(2)
