from manim import *
import numpy as np

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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "A factorial is the product of descending positive integers.",
            "Sammy the Squirrel has three different colored acorns.",
            "He can arrange them in many distinct sequences.",
            "Three times two times one equals six total ways.",
            "This pattern works perfectly for all positive numbers."
        ]
        
        self.setup_layout("Prerequisite: The Rule of n!", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # General factorial definition - Replaced MathTex with Text to avoid LaTeX dependency
        gen_formula = Text("n! = n × (n-1) × ... × 1", color=WHITE)
        self.place_in_area(gen_formula, "B2", "B5", scale_factor=0.9)
        self.play(Write(gen_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            FadeOut(gen_formula)
        )

        # Sammy and Acorns (Icons)
        try:
            sammy = ImageMobject("Sammy.png")
        except:
            sammy = Text("Sammy", color=ORANGE, font_size=24)
            
        self.place_at_grid(sammy, "C1", scale_factor=1.2)
        
        red_icon = Circle(radius=0.4, fill_opacity=1, color="#FF0000", stroke_width=0)
        green_icon = Square(side_length=0.7, fill_opacity=1, color="#00FF00", stroke_width=0)
        blue_icon = Triangle(fill_opacity=1, color="#0000FF", stroke_width=0).scale(0.5)

        self.place_at_grid(red_icon, "D2")
        self.place_at_grid(green_icon, "D3")
        self.place_at_grid(blue_icon, "D4")

        self.play(FadeIn(sammy), FadeIn(red_icon), FadeIn(green_icon), FadeIn(blue_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Permutations shuffling
        pos_indices = ["D2", "D3", "D4"]
        perms = [
            (red_icon, green_icon, blue_icon),
            (red_icon, blue_icon, green_icon),
            (green_icon, red_icon, blue_icon),
            (green_icon, blue_icon, red_icon),
            (blue_icon, red_icon, green_icon),
            (blue_icon, green_icon, red_icon),
        ]

        for i, perm in enumerate(perms):
            self.play(
                perm[0].animate.move_to(self.grid[pos_indices[0]]),
                perm[1].animate.move_to(self.grid[pos_indices[1]]),
                perm[2].animate.move_to(self.grid[pos_indices[2]]),
                run_time=0.3
            )
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )

        # 3! Formula - Replaced MathTex with Text to avoid LaTeX dependency
        formula_3 = Text("3! = 3 × 2 × 1 = 6", color=WHITE)
        self.place_in_area(formula_3, "A2", "A5", scale_factor=1.0)
        self.play(Write(formula_3))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        self.wait(2)
        
        # Cleanup for next section
        self.play(
            FadeOut(sammy), 
            FadeOut(red_icon), 
            FadeOut(green_icon), 
            FadeOut(blue_icon), 
            FadeOut(formula_3),
            self.lecture[4].animate.set_color(WHITE)
        )
