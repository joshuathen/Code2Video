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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Vectors model motion and space.", "They power physics engines.", "Fundamental to modern navigation."]
        self.setup_layout("Summary & Real-World Application", lecture_lines)
        
        COLOR_1 = "#FFFFFF"
        COLOR_2 = "#FFFF00"
        COLOR_3 = "#00FFFF"

        # Asset loading
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg")
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")

        # === Animation for Lecture Line 1 ===
        vectors = VGroup(*[
            Vector(direction=RIGHT*0.5 + UP*0.5, color=COLOR_1),
            Vector(direction=LEFT*0.3 + UP*0.6, color=COLOR_1),
            Vector(direction=DOWN*0.4 + RIGHT*0.2, color=COLOR_1),
        ])
        self.place_at_grid(vectors, 'A2', scale_factor=0.8)
        
        arrow_1 = Arrow(start=ORIGIN, end=RIGHT, color=COLOR_1)
        self.place_at_grid(arrow_1, 'A1', scale_factor=0.5)
        
        label_vectors = Text("Set of Vectors", font_size=20, color=COLOR_1)
        label_vectors.next_to(vectors, UP, buff=0.1)

        self.play(Create(vectors), Create(arrow_1), Write(label_vectors), FadeIn(satellite.scale(0.3).next_to(vectors, RIGHT)))
        self.lecture[0].set_color(COLOR_1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        rotation_label = Text("Rotation", font_size=20, color=COLOR_2)
        self.place_at_grid(rotation_label, 'B2', scale_factor=0.8)
        
        arrow_2 = Arrow(start=ORIGIN, end=RIGHT, color=COLOR_2)
        self.place_at_grid(arrow_2, 'B1', scale_factor=0.5)
        
        self.play(
            Rotate(vectors[0], angle=PI/2, about_point=vectors[0].get_start()),
            FadeIn(rotation_label),
            FadeIn(arrow_2),
            self.lecture[1].animate.set_color(COLOR_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        system_label = Text("System", font_size=20, color=COLOR_3)
        self.place_at_grid(system_label, 'C2', scale_factor=0.8)
        
        arrow_3 = Arrow(start=ORIGIN, end=RIGHT, color=COLOR_3)
        self.place_at_grid(arrow_3, 'C1', scale_factor=0.5)
        
        self.play(
            vectors.animate.set_color(COLOR_3),
            FadeIn(system_label),
            FadeIn(arrow_3),
            FadeIn(robot.scale(0.3).next_to(vectors, LEFT)),
            self.lecture[2].animate.set_color(COLOR_3)
        )
        self.wait(2)
