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

class Section1Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout with title and lecture lines
        title = "The Hook: The Alchemist's Dilemma"
        lines = [
            "Meet Leo and his two random magic jars.",
            "Jar A drops X coins; Jar B drops Y coins.",
            "We need the probability that their sum equals four."
        ]
        self.setup_layout(title, lines)
        
        # Colors from storyboard and visual logic
        COLOR_A = "#ADD8E6"
        COLOR_B = "#FFD700"
        COLOR_POTION = "#FF69B4"
        COLOR_TEXT = "#FFFFFF"

        # Asset path
        JAR_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/jar.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line and introduce magic jars
        self.play(self.lecture[0].animate.set_color(COLOR_A))
        
        # Jar icons using the provided SVGMobject asset
        jar_a = SVGMobject(JAR_ASSET).set_color(COLOR_A)
        jar_b = SVGMobject(JAR_ASSET).set_color(COLOR_B)
        
        # Jar labels positioned above the icons
        label_jar_a = Text("Jar A", font_size=20, color=COLOR_A)
        label_jar_b = Text("Jar B", font_size=20, color=COLOR_B)
        
        # Position jars on the grid (B2 and B5 for separation)
        self.place_at_grid(jar_a, "B2", scale_factor=0.8)
        self.place_at_grid(jar_b, "B5", scale_factor=0.8)
        
        # Labels within 1 grid unit (Proximity Rule L002)
        label_jar_a.next_to(jar_a, UP, buff=0.1)
        label_jar_b.next_to(jar_b, UP, buff=0.1)
        
        self.play(FadeIn(jar_a), FadeIn(jar_b), Write(label_jar_a), Write(label_jar_b))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Highlight second line and show X, Y drops and the target potion
        self.play(self.lecture[1].animate.set_color(COLOR_B))
        
        # Variables X and Y for the drops, below the jars
        var_x = Text("X coins", font_size=20, color=COLOR_A)
        var_y = Text("Y coins", font_size=20, color=COLOR_B)
        var_x.next_to(jar_a, DOWN, buff=0.2)
        var_y.next_to(jar_b, DOWN, buff=0.2)
        
        # Potion bottle representation using simple geometric shapes
        potion_body = Triangle(color=COLOR_POTION).set_fill(COLOR_POTION, opacity=0.3)
        potion_cork = Square(side_length=0.15, color=COLOR_POTION).set_fill(COLOR_POTION, opacity=1).next_to(potion_body, UP, buff=0)
        potion_label = Text("4 Coins", font_size=16, color=WHITE).move_to(potion_body.get_center() + DOWN*0.1)
        potion = VGroup(potion_body, potion_cork, potion_label)
        
        # Position potion in lower central area (E3-E4) to avoid clutter
        self.place_in_area(potion, "E3", "E4", scale_factor=0.9)
        
        self.play(FadeIn(potion), Write(var_x), Write(var_y))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight third line and show the sum question text
        self.play(self.lecture[2].animate.set_color(COLOR_POTION))
        
        # Question text Centered between jars and potion
        sum_text = Text("Jar A + Jar B = ?", font_size=24, color=COLOR_TEXT)
        # Use place_in_area for multi-word label centering (L002)
        self.place_in_area(sum_text, "B3", "C4", scale_factor=0.8)
        
        self.play(FadeIn(sum_text))
        self.wait(2.0)
