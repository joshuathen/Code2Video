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
        title = "The Hook: The Detective Dog's Dilemma"
        lecture_lines = [
            "Meet Sherlock Bones, our probabilistic detective dog.",
            "He hears a rustle in the bushes nearby.",
            "Sherlock believes there's a 10% chance it's a cat.",
            "If it's a cat, his ears twitch 90% often.",
            "His ears just twitched! Is it really a cat?"
        ]
        
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_PRIOR = "#FFFF00"
        COLOR_DOG = "#C0C0C0"
        COLOR_CAT = "#FF69B4"
        COLOR_WIND = "#00FFFF"
        COLOR_BUSH = "#4F7942"

        # === Animation for Lecture Line 1 ===
        # Meet Sherlock Bones (Dog Icon Asset)
        dog = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dog.svg").set_color(COLOR_DOG)
        # Fix Issue 33: Position dog at D4
        self.place_at_grid(dog, "D4", scale_factor=0.8)

        self.play(self.lecture[0].animate.set_color(COLOR_DOG))
        self.play(FadeIn(dog))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rustle in the bushes
        bush = VGroup(*[
            Circle(radius=0.3, color=COLOR_BUSH, fill_opacity=1).shift(RIGHT*0.2*i)
            for i in range(3)
        ])
        self.place_at_grid(bush, "D6", scale_factor=0.8)
        
        self.play(self.lecture[1].animate.set_color(COLOR_BUSH))
        self.play(FadeIn(bush))
        # Shake animation
        for _ in range(2):
            self.play(bush.animate.shift(LEFT*0.1), run_time=0.1)
            self.play(bush.animate.shift(RIGHT*0.1), run_time=0.1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Sherlock believes 10% Cat / 90% Wind (Thought Bubble with Asset)
        bubble_outline = RoundedRectangle(corner_radius=0.2, height=1.4, width=2.4, color=WHITE)
        
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png").scale(0.2)
        cat_text = Text("10%", font_size=18, color=WHITE)
        # Fixed: Use Group instead of VGroup for ImageMobject
        cat_group = Group(cat_icon, cat_text).arrange(RIGHT, buff=0.1)
        
        wind_text = Text("Wind (90%)", font_size=18, color=WHITE)
        # Fixed: Use Group for consistency with cat_group
        thought_content = Group(cat_group, wind_text).arrange(DOWN, buff=0.2)
        
        # Fixed: Use Group because it contains ImageMobject indirectly
        thought_bubble = Group(bubble_outline, thought_content)
        # Fix Issue 33: Position thought_bubble at C5
        self.place_at_grid(thought_bubble, "C5", scale_factor=1.0)
        
        # Small dots connecting dog to bubble
        connector_dots = VGroup(
            Dot(radius=0.04, color=WHITE),
            Dot(radius=0.06, color=WHITE),
            Dot(radius=0.08, color=WHITE)
        ).arrange(UP+RIGHT, buff=0.1)
        connector_dots.next_to(dog, UP+RIGHT, buff=-0.2)

        self.play(self.lecture[2].animate.set_color(COLOR_PRIOR))
        self.play(Create(connector_dots), FadeIn(thought_bubble))
        # Highlight Cat as Prior
        self.play(cat_text.animate.set_color(COLOR_PRIOR), cat_icon.animate.set_color(COLOR_PRIOR))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # If it's a cat, his ears twitch 90% often
        twitch_prob = Text("P(Twitch | Cat) = 90%", font_size=16, color=COLOR_CAT)
        # Fix Issue 34: Position twitch_prob at B5
        self.place_at_grid(twitch_prob, "B5", scale_factor=1.0)

        self.play(self.lecture[3].animate.set_color(COLOR_CAT))
        self.play(FadeIn(twitch_prob))
        
        # Twitching animation on the dog icon (rotation)
        # Since it's an SVG, we just rotate the whole icon slightly back and forth
        for _ in range(2):
            self.play(dog.animate.rotate(10*DEGREES), run_time=0.1)
            self.play(dog.animate.rotate(-20*DEGREES), run_time=0.1)
            self.play(dog.animate.rotate(10*DEGREES), run_time=0.1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # His ears just twitched! Is it really a cat?
        question = Text("P(Cat | Twitch)?", font_size=24, color=YELLOW)
        # Fix Issue 35: Position question at E4
        self.place_at_grid(question, "E4", scale_factor=1.2)

        self.play(self.lecture[4].animate.set_color(YELLOW))
        
        # Final twitch
        self.play(dog.animate.rotate(10*DEGREES), run_time=0.05)
        self.play(dog.animate.rotate(-20*DEGREES), run_time=0.05)
        self.play(dog.animate.rotate(10*DEGREES), run_time=0.05)
        
        self.play(Write(question))
        self.wait(2)
