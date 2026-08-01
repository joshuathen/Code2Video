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

class Section3Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "The Sampling Experiment"
        lecture_lines = [
            "Let's take random groups, or samples, of monsters.",
            "For each group, we calculate the average weight.",
            "This average is called the \"Sample Mean.\"",
            "We repeat this process many, many times.",
            "Each sample gives us one new data point."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors defined in storyboard
        LIME = "#32CD32"
        GOLD = "#FFD700"
        WHITE_CLR = "#FFFFFF"

        # Assets
        MONSTER_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/monster.svg"

        # === Animation for Lecture Line 1 ===
        # "Let's take random groups, or samples, of monsters."
        self.play(self.lecture[0].animate.set_color(LIME))
        
        # Create monster icons using the provided asset
        # Integration of Issue 21: Use [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/monster.svg]
        monsters = VGroup(*[
            SVGMobject(MONSTER_ASSET, color=LIME, fill_opacity=1).scale(0.2)
            for _ in range(30)
        ])
        monsters.arrange_in_grid(rows=5, cols=6, buff=0.15)
        
        # Integration of Issue 28: Adjust position to B2-E4 and scale to 0.8
        self.place_in_area(monsters, "B2", "E4", scale_factor=0.8)
        
        self.play(FadeIn(monsters))
        # Flickering effect (simulation of sampling)
        self.play(
            LaggedStart(*[
                Indicate(m, color=LIME, scale_factor=1.1) 
                for m in monsters
            ], lag_ratio=0.03),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "For each group, we calculate the average weight."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GOLD)
        )
        
        # Integration of Issue 29: Adjust formula position to A3-A6 and scale to 0.8
        formula = MathTex(r"\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i", color=GOLD)
        self.place_in_area(formula, "A3", "A6", scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This average is called the \"Sample Mean.\""
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GOLD)
        )
        
        label = Text("Sample Mean", font_size=24, color=GOLD)
        # Position label relative to formula but ensure it stays within 1 grid unit
        label.next_to(formula, UP, buff=0.2)
        
        self.play(FadeIn(label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "We repeat this process many, many times."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(WHITE_CLR)
        )
        
        # Integration of Issue 30: Adjust means_list scale to 0.7 at B5-E6
        means_list = VGroup(*[
            MathTex(fr"\bar{{X}}_{{{i+1}}}", color=WHITE_CLR) 
            for i in range(5)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        self.place_in_area(means_list, "B5", "E6", scale_factor=0.7)
        
        # Sampling animation loop showing new data points appearing
        # We use a subset of monsters to represent a "sample" being processed
        for i in range(5):
            # Highlight a "random" subset of monsters to represent a new sample
            indices = np.random.choice(range(30), 6, replace=False)
            subset = VGroup(*[monsters[idx] for idx in indices])
            
            # Pulse the subset and then show the mean result
            self.play(
                subset.animate.set_color(GOLD),
                run_time=0.3
            )
            self.play(
                FadeIn(means_list[i]),
                subset.animate.set_color(LIME),
                run_time=0.3
            )
            
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Each sample gives us one new data point."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE_CLR)
        )
        
        self.play(Indicate(means_list, color=WHITE_CLR))
        self.wait(2)
