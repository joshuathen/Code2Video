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
        self.setup_layout("The Big Picture: Why We Solve Them", [
            "PDEs describe the fundamental language of nature.",
            "Simulate airflow for efficient airplane design.",
            "Predict weather and analyze complex signals."
        ])
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Wave propagation placeholder
        wave = VGroup(*[Line(LEFT*0.5, RIGHT*0.5).shift(UP*i*0.2) for i in range(-2, 3)])
        wave.set_color(WHITE)
        self.place_at_grid(wave, 'B2', scale_factor=1.0)
        self.play(self.lecture[0].animate.set_color("#3357FF"), wave.animate.set_color("#3357FF"))
        
        # === Animation for Lecture Line 2 ===
        # Airplane asset
        airplane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg")
        airplane.set_color(WHITE)
        self.place_in_area(airplane, 'C5', 'D6', scale_factor=0.8)
        self.play(self.lecture[1].animate.set_color("#FF5733"), airplane.animate.set_color("#FF5733"))
        
        # === Animation for Lecture Line 3 ===
        # Cloud asset
        cloud = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg")
        cloud.set_color(WHITE)
        self.place_at_grid(cloud, 'E5', scale_factor=0.7)
        self.play(self.lecture[2].animate.set_color("#33FF57"), cloud.animate.set_color("#33FF57"))
        
        self.wait(2)
