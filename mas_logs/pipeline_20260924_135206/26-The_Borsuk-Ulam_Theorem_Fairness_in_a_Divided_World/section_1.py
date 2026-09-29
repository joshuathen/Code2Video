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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Continuous mapping links spheres to flat spaces.", "Imagine stretching a globe onto a plane.", "We must do this without any tearing."]
        self.setup_layout("Prerequisite: The Concept of Continuous Mapping", lecture_lines)
        
        # Animation Objects
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg", color="#FFFFFF")
        label_s = MathTex("S^n", color="#FF00FF")
        plane = Square(side_length=3, color="#00FFFF").set_opacity(0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(globe, 'A3', 'C6', scale_factor=0.6)
        label_s.next_to(globe, UP)
        self.play(Create(globe), Write(label_s))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5733"))
        target_plane = self.place_in_area(plane.copy(), 'D3', 'F6', scale_factor=0.6)
        self.play(Transform(globe, target_plane))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33FF57"))
        flash = Circle(radius=1.6, color="#33FF57").set_stroke(width=4)
        self.place_at_grid(flash, 'F3', scale_factor=0.5)
        self.play(Flash(flash.get_center(), color="#33FF57", line_length=0.2, num_lines=15))
        self.wait(1)
