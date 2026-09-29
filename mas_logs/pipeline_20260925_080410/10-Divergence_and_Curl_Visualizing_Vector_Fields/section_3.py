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
        self.setup_layout("Curl: The Rotation Factor", [
            "Curl measures microscopic rotation at points.",
            "A spinning paddlewheel indicates non-zero curl.",
            "The curl vector points along rotation axes."
        ])
        
        # Paddlewheel representation
        # Use C4 as per issue 25
        wheel = VGroup(*[Line(ORIGIN, 0.8 * UP).rotate(i * 360/8 * DEGREES, about_point=ORIGIN) for i in range(8)])
        wheel.set_stroke(color="#FFA500", width=4)
        self.place_at_grid(wheel, "C4", scale_factor=1.0)
        
        curl_label = Text("Curl", font_size=30, color="#FF00FF")
        # Place at D3 as per issue 27
        self.place_at_grid(curl_label, "D3", scale_factor=0.8)
        curl_label.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFA500"))
        self.play(Create(wheel))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFA500"))
        self.play(Rotate(wheel, angle=2*PI, run_time=2))
        self.play(Rotate(wheel, angle=-2*PI, run_time=2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        self.play(FadeIn(curl_label))
        self.wait(2)
