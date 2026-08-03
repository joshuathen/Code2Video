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
        self.setup_layout(
            "The Intuition: Why Focus Matters", 
            [
                "We don't process every detail in a busy scene.",
                "We selectively focus on what is most relevant.",
                "Attention in AI mimics this human cognitive focus."
            ]
        )
        
        # Colors
        GRAY_DIM = "#808080"
        WHITE_CLR = "#FFFFFF"
        YELLOW_CLR = "#FFFF00"
        RED_CLR = "#FF0000"
        GREEN_CLR = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(WHITE_CLR))

        # Create scattered icons (simple shapes representing leaves, flowers, rocks)
        icons = VGroup()
        # Leaves (Triangles)
        for pos in ["A2", "B4", "D1", "E5"]:
            leaf = Triangle(color=GRAY_DIM, fill_opacity=0.5).scale(0.15)
            self.place_at_grid(leaf, pos)
            icons.add(leaf)
        # Flowers (Circles)
        for pos in ["B2", "C5", "F1", "D6"]:
            flower = Circle(color=GRAY_DIM, fill_opacity=0.5).scale(0.15)
            self.place_at_grid(flower, pos)
            icons.add(flower)
        # Rocks (Squares)
        for pos in ["A6", "C1", "E3", "F4"]:
            rock = Square(color=GRAY_DIM, fill_opacity=0.5).scale(0.15)
            self.place_at_grid(rock, pos)
            icons.add(rock)

        self.play(FadeIn(icons))

        # Spotlight logic
        spotlight = Circle(radius=0.8, color=WHITE_CLR, stroke_width=2).set_fill(WHITE_CLR, opacity=0.1)
        # Issue 17: Fixed clipping by moving initial position to B2
        self.place_at_grid(spotlight, "B2")
        
        self.play(Create(spotlight))
        # Move spotlight across icons
        self.play(spotlight.animate.move_to(self.grid["F6"]), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight
        self.play(
            self.lecture[0].animate.set_color(GRAY_DIM),
            self.lecture[1].animate.set_color(YELLOW_CLR)
        )

        # Cat icon (Yellow)
        cat_body = Circle(radius=0.2, color=YELLOW_CLR, fill_opacity=1)
        cat_ear1 = Triangle(color=YELLOW_CLR, fill_opacity=1).scale(0.05).rotate(30*DEGREES).move_to(cat_body.get_top() + LEFT*0.1)
        cat_ear2 = Triangle(color=YELLOW_CLR, fill_opacity=1).scale(0.05).rotate(-30*DEGREES).move_to(cat_body.get_top() + RIGHT*0.1)
        cat = VGroup(cat_body, cat_ear1, cat_ear2)
        self.place_at_grid(cat, "C3")

        # Mouse icon (Red)
        mouse = Circle(radius=0.1, color=RED_CLR, fill_opacity=1)
        self.place_at_grid(mouse, "D4")

        # Spotlight on the mouse
        self.play(
            FadeIn(cat),
            FadeIn(mouse),
            spotlight.animate.move_to(self.grid["D4"]).scale(0.8),
        )

        # Mouse pulses
        self.play(
            mouse.animate.scale(1.5),
            rate_func=there_and_back,
            run_time=1
        )

        # Fade out other icons
        self.play(icons.animate.set_opacity(0.1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight
        self.play(
            self.lecture[1].animate.set_color(GRAY_DIM),
            self.lecture[2].animate.set_color(GREEN_CLR)
        )

        # Clear previous icons and spotlight
        self.play(
            FadeOut(icons),
            FadeOut(cat),
            FadeOut(mouse),
            FadeOut(spotlight)
        )

        # Word indices in Text object (simple way: build it from words)
        words = VGroup(*[Text(w, font_size=32, color=GRAY_DIM) for w in ["The", "bank", "of", "the", "river"]])
        words.arrange(RIGHT, buff=0.2)
        # Issue 18: Fixed crowding by moving from C1-C6 to C2-C6 and scaling down
        self.place_in_area(words, "C2", "C6", scale_factor=0.8)

        self.play(Write(words))
        self.wait(0.5)

        # Brighten 'bank' and 'river'
        self.play(
            words[1].animate.set_color(WHITE_CLR), # bank
            words[4].animate.set_color(WHITE_CLR)  # river
        )

        # Connecting line
        conn_line = Line(
            words[1].get_bottom(), 
            words[4].get_bottom(), 
            color=GREEN_CLR,
            buff=0.1
        ).add_tip(tip_length=0.15)
        
        # Label 'Landform'
        label = Text("Landform", font_size=24, color=GREEN_CLR)
        label.next_to(words[1], UP, buff=0.3)

        self.play(
            Create(conn_line),
            FadeIn(label)
        )
        
        self.wait(2)
