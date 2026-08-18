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
        self.setup_layout("Summary and Conclusion", [
            "Light changes direction when speed changes.",
            "Refraction depends on the refractive index.",
            "The normal line is our reference point."
        ])
        
        # Animation Elements
        light_ray = Line(start=ORIGIN, end=RIGHT*2, color=YELLOW)
        normal_line = DashedLine(start=UP*2, end=DOWN*2, color=WHITE)
        quiz_text = Text("If light enters a medium\nwith a higher n, does it bend\ntoward or away from normal?", font_size=24, color=BLUE)
        answer_text = Text("Answer: Toward", font_size=32, color=GREEN)
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(light_ray, 'B2', scale_factor=1.0)
        self.play(Create(light_ray))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.place_in_area(quiz_text, 'D2', 'D4', scale_factor=0.6)
        self.play(FadeIn(quiz_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        self.place_at_grid(normal_line, 'C5', scale_factor=0.9)
        self.play(Create(normal_line))
        self.place_at_grid(answer_text, 'E5', scale_factor=0.7)
        self.play(Write(answer_text))
        
        self.place_at_grid(prism, 'F3', scale_factor=1.0)
        self.play(FadeIn(prism))
        self.wait(2)
