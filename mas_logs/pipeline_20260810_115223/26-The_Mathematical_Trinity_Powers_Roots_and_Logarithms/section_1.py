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
        lecture_lines = [
            "Imagine a seed growing into a tree.",
            "The base is our starting seed.",
            "The exponent is our growth time.",
            "The result is the final height.",
            "Two raised to three equals eight. [Asset: growth_tree_animation]"
        ]
        self.setup_layout("The Bridge: Prerequisite Recap", lecture_lines)
        
        seed = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seed.svg")
        exponent = Text("t", color="#FFCC00")
        result = Text("8", color="#FF0000")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeOut(self.title))
        self.place_in_area(seed, 'B2', 'B4', scale_factor=1.2)
        self.play(FadeIn(seed.set_color("#00FF00")))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(exponent, 'B5', scale_factor=1.0)
        self.play(FadeIn(exponent))
        self.lecture[2].set_color("#00FF00")

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(result, 'C5', scale_factor=1.0)
        self.play(Write(result))
        self.lecture[3].set_color("#00FF00")

        # === Animation for Lecture Line 5 ===
        # Re-using the seed for the emphasis
        self.play(Flash(result, color="#FF0000"), Flash(seed, color="#FF0000"))
        
        tree = Square(color=WHITE) # Placeholder for [Asset: growth_tree_animation]
        self.place_in_area(tree, 'C2', 'D5', scale_factor=0.9)
        self.play(Create(tree))
        
        caption_label = Text("2^3=8", color=WHITE)
        self.place_at_grid(caption_label, 'F2', scale_factor=0.7)
        self.play(FadeIn(caption_label))
        
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
