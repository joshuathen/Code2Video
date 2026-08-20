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
        lecture_lines = ["Classical bits are strictly zero or one.", "Qubits leverage superposition to be both.", "This enables massive parallel information processing."]
        self.setup_layout("Application: Quantum Computing Qubits", lecture_lines)
        
        # Asset path
        sphere_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        
        # === Animation for Lecture Line 1 ===
        sphere = SVGMobject(sphere_asset, color="#00AAFF")
        self.place_at_grid(sphere, "C4", scale_factor=0.8)
        self.play(FadeIn(sphere))
        self.play(self.lecture[0].animate.set_color("#00AAFF"))

        # === Animation for Lecture Line 2 ===
        state_vector = Arrow(start=ORIGIN, end=UP*1.0, color="#FFFFFF", buff=0)
        group = VGroup(sphere, state_vector)
        self.place_in_area(group, 'B2', 'D4', scale_factor=0.95)
        self.play(Create(state_vector))
        self.play(Rotate(state_vector, angle=2*PI, axis=RIGHT, about_point=sphere.get_center()), run_time=2)
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 3 ===
        pole_0 = Dot(point=sphere.get_center() + UP*1.0, color="#FFFF00")
        pole_1 = Dot(point=sphere.get_center() + DOWN*1.0, color="#FFFF00")
        label_0 = Text("0", color="#FFFF00", font_size=20).next_to(pole_0, UP, buff=0.1)
        label_1 = Text("1", color="#FFFF00", font_size=20).next_to(pole_1, DOWN, buff=0.1)
        
        self.play(Create(pole_0), Create(pole_1), Write(label_0), Write(label_1))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(1)
