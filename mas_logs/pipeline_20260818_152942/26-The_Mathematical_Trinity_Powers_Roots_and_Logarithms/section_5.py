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
        self.setup_layout("Summary & Quick Check", [
            "Fuel is the exponent power.",
            "Distance is the root calculation.",
            "Timing is the log requirement."
        ])
        
        # Rocket setup using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg]
        rocket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg", color=WHITE)
        rocket_label = Text("Base 10", font_size=18, color=WHITE)
        rocket_group = VGroup(rocket, rocket_label).arrange(DOWN)
        
        # Applying critique fixes
        self.place_at_grid(rocket_group, 'B2', scale_factor=0.8)
        self.play(FadeIn(rocket_group))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        fuel_text = MathTex("10^2 = 100", color="#FFD700")
        self.place_at_grid(fuel_text, 'C4', scale_factor=0.9)
        self.play(Write(fuel_text))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        dist_text = MathTex("\\sqrt{100} = 10", color="#00FF00")
        self.place_at_grid(dist_text, 'D4', scale_factor=0.9)
        self.play(Write(dist_text))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        # Rocket reused for log
        rocket2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg", color="#FF4500")
        log_text = MathTex("\\log_{10}(100) = 2", color="#FF4500")
        self.place_at_grid(log_text, 'E4', scale_factor=0.9)
        self.place_at_grid(rocket2, 'E2', scale_factor=0.6)
        self.play(Write(log_text), FadeIn(rocket2))
        self.wait(2)
