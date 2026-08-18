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
        self.setup_layout("The Duality Bridge", ["Dual connects physical things to measurements.", "Symmetry links vectors and scalar functions.", "This bridge enables physics and AI."])
        
        # Elements
        vector_label = Text("Vector (Thing)", color=BLUE, font_size=24)
        function_label = Text("Functional (Measurement)", color=YELLOW, font_size=24)
        bridge = Arrow(start=ORIGIN, end=RIGHT*2, color=WHITE)
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(vector_label, 'B2', 'B2', scale_factor=0.7)
        self.place_at_grid(bridge, 'B3', scale_factor=0.7)
        self.place_in_area(function_label, 'B4', 'B4', scale_factor=0.7)
        self.place_at_grid(sensor_icon, 'A3', scale_factor=0.5)
        self.play(FadeIn(vector_label), GrowArrow(bridge), FadeIn(function_label), FadeIn(sensor_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        symmetry_text = Text("Symmetry: v ↔ f(v)", color=WHITE, font_size=24)
        self.place_at_grid(symmetry_text, 'C3', scale_factor=0.9)
        self.play(Write(symmetry_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        apps = VGroup(
            Text("Physics", color=GREEN, font_size=20),
            Text("Optimization", color=GREEN, font_size=20),
            Text("AI / Neural Nets", color=GREEN, font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(apps, 'D2', 'E4', scale_factor=0.8)
        self.place_at_grid(computer_icon, 'F3', scale_factor=0.5)
        self.play(FadeIn(apps), FadeIn(computer_icon))
        self.wait(2)
