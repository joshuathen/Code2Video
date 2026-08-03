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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Prerequisite: Measuring Similarity with Vectors",
            [
                "Words can be represented as arrows called vectors.",
                "Directions indicate the meaning and relationship of words.",
                "Dot products measure how much these vectors align."
            ]
        )
        
        green_color = "#90EE90"  # Light green for 'Apple' and 'Fruit'
        pink_color = "#FFB6C1"   # Light pink for 'Apple' and 'Hammer'

        # Load assets
        apple_svg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/apple.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create Vector Apple (Green)
        v_apple_g = Arrow(ORIGIN, RIGHT * 1.2, color=green_color, stroke_width=4, buff=0)
        apple_icon_g = SVGMobject(apple_svg_path).set_color(green_color).set_height(0.3)
        apple_label_g = Text("Apple", font_size=20, color=green_color)
        l_apple_g = VGroup(apple_icon_g, apple_label_g).arrange(RIGHT, buff=0.1)
        
        self.place_at_grid(v_apple_g, "B2")
        self.place_at_grid(l_apple_g, "A2")
        
        # Create Vector Fruit (Green)
        v_fruit_g = Arrow(ORIGIN, RIGHT * 1.2, color=green_color, stroke_width=4, buff=0)
        l_fruit_g = Text("Fruit", font_size=20, color=green_color)
        
        # Applying Issue 38 Fix: v_fruit_g -> 'B5', l_fruit_g -> 'A5'
        self.place_at_grid(v_fruit_g, "B5")
        self.place_at_grid(l_fruit_g, "A5")
        
        self.play(Create(v_apple_g), FadeIn(l_apple_g))
        self.play(Create(v_fruit_g), Write(l_fruit_g))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Align Apple and Fruit (similar meaning)
        self.play(
            v_apple_g.animate.set_angle(PI/6),
            v_fruit_g.animate.set_angle(PI/6 + 0.1)
        )
        self.wait(1)
        
        # Introduce Apple and Hammer comparison (different meaning)
        v_apple_p = Arrow(ORIGIN, RIGHT * 1.2, color=pink_color, stroke_width=4, buff=0)
        apple_icon_p = SVGMobject(apple_svg_path).set_color(pink_color).set_height(0.3)
        apple_label_p = Text("Apple", font_size=20, color=pink_color)
        l_apple_p = VGroup(apple_icon_p, apple_label_p).arrange(RIGHT, buff=0.1)

        v_hammer_p = Arrow(ORIGIN, RIGHT * 1.2, color=pink_color, stroke_width=4, buff=0)
        l_hammer_p = Text("Hammer", font_size=20, color=pink_color)

        # Applying Issue 38 and 39 Fixes:
        # l_apple_p -> 'C2', v_apple_p -> 'D2'
        # l_hammer_p -> 'C5', v_hammer_p -> 'D5'
        self.place_at_grid(v_apple_p, "D2")
        self.place_at_grid(l_apple_p, "C2")
        self.place_at_grid(v_hammer_p, "D5")
        self.place_at_grid(l_hammer_p, "C5")

        # Fade out Green slightly and fade in Pink set
        self.play(
            v_apple_g.animate.set_opacity(0.3),
            l_apple_g.animate.set_opacity(0.3),
            v_fruit_g.animate.set_opacity(0.3),
            l_fruit_g.animate.set_opacity(0.3),
            FadeIn(v_apple_p), FadeIn(l_apple_p),
            FadeIn(v_hammer_p), FadeIn(l_hammer_p)
        )
        
        # Orient Apple and Hammer in opposite directions
        self.play(
            v_apple_p.animate.set_angle(0),
            v_hammer_p.animate.set_angle(PI - 0.1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Define Similarity Counter
        sim_label = Text("Similarity Score:", font_size=22)
        sim_score = DecimalNumber(0.0, num_decimal_places=2, color=YELLOW)
        counter_group = VGroup(sim_label, sim_score).arrange(RIGHT)
        
        # Applying Issue 40 Fix: Move to 'E3'
        self.place_at_grid(counter_group, "E3", scale_factor=0.9)
        
        self.play(Write(counter_group))
        
        # First: Show High similarity for Green vectors
        self.play(
            v_apple_g.animate.set_opacity(1),
            l_apple_g.animate.set_opacity(1),
            v_fruit_g.animate.set_opacity(1),
            l_fruit_g.animate.set_opacity(1),
            v_apple_p.animate.set_opacity(0.2),
            l_apple_p.animate.set_opacity(0.2),
            v_hammer_p.animate.set_opacity(0.2),
            l_hammer_p.animate.set_opacity(0.2),
        )
        self.play(ChangeDecimalToValue(sim_score, 0.95), run_time=1.5)
        self.wait(1)
        
        # Second: Show Low similarity for Pink vectors
        self.play(
            v_apple_g.animate.set_opacity(0.2),
            l_apple_g.animate.set_opacity(0.2),
            v_fruit_g.animate.set_opacity(0.2),
            l_fruit_g.animate.set_opacity(0.2),
            v_apple_p.animate.set_opacity(1),
            l_apple_p.animate.set_opacity(1),
            v_hammer_p.animate.set_opacity(1),
            l_hammer_p.animate.set_opacity(1),
        )
        self.play(ChangeDecimalToValue(sim_score, -0.88), run_time=1.5)
        self.wait(2)
