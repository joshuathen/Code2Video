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
        title = "The Library Analogy: Query, Key, and Value"
        lines = [
            "The Query represents the word looking for context.",
            "Keys are labels used to find matches.",
            "Matching labels unlock the word's hidden Value.",
            "Similarity scores determine how much information to take.",
            "This database search forms the core of attention."
        ]
        self.setup_layout(title, lines)
        
        # Color constants
        COLOR_Q = "#00FFFF"
        COLOR_K = "#F0E68C"
        COLOR_V = "#ADFF2F"

        # === Animation for Lecture Line 1 ===
        # The Query represents the word looking for context.
        self.play(self.lecture[0].animate.set_color(COLOR_Q))
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magn.svg
        query_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magn.svg").set_color(COLOR_Q)
        query_label = Text("What we want", font_size=18, color=COLOR_Q).next_to(query_icon, UP, buff=0.1)
        query_group = VGroup(query_icon, query_label)
        
        # Resolved Issue 25: Positioning at B2, scale 0.8
        self.place_at_grid(query_group, "B2", scale_factor=0.8)
        self.play(FadeIn(query_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Keys are labels used to find matches.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_K)
        )
        
        keys = VGroup()
        key_labels = ["K1", "K2", "K3", "K4"]
        grid_positions = ["D2", "D3", "D4", "D5"]
        
        for i in range(4):
            folder_body = RoundedRectangle(corner_radius=0.05, height=0.5, width=0.7, color=COLOR_K, fill_opacity=0.3)
            folder_tab = Rectangle(height=0.1, width=0.25, color=COLOR_K, fill_opacity=0.3).next_to(folder_body, UP, buff=0, aligned_edge=LEFT)
            folder = VGroup(folder_body, folder_tab)
            k_label = Text(key_labels[i], font_size=16, color=COLOR_K).move_to(folder_body.get_center())
            key_unit = VGroup(folder, k_label)
            self.place_at_grid(key_unit, grid_positions[i])
            keys.add(key_unit)
            
        self.play(Create(keys))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matching labels unlock the word's hidden Value.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_V)
        )
        
        # Query icon moving past Keys, pausing at each
        for i in range(4):
            target_pos = self.grid[grid_positions[i]] + UP * 0.7
            self.play(query_group.animate.move_to(target_pos), run_time=0.6)
            
            # Flash effect to indicate comparison
            flash = Circle(radius=0.1, color=WHITE).move_to(target_pos).set_opacity(0.5)
            self.play(flash.animate.scale(3).set_opacity(0), run_time=0.3)
            self.remove(flash)
        
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Similarity scores determine how much information to take.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        scores_vals = ["0.12", "0.85", "0.02", "0.01"]
        score_labels = VGroup()
        score_positions = ["C2", "C3", "C4", "C5"]
        
        for i in range(4):
            score_txt = Text(scores_vals[i], font_size=20, color=YELLOW)
            self.place_at_grid(score_txt, score_positions[i])
            score_labels.add(score_txt)
            
        self.play(Write(score_labels))
        self.play(Indicate(score_labels[1], color=GOLD, scale_factor=1.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This database search forms the core of attention.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_V)
        )
        
        # Open highest-scoring folder (Keys[1]) to reveal Value
        value_box = Rectangle(height=0.4, width=0.6, color=COLOR_V, fill_opacity=0.6)
        value_text = Text("Value", font_size=18, color=BLACK).move_to(value_box.get_center())
        value_group = VGroup(value_box, value_text)
        
        # Resolved Issue 26: Positioning at E4, scale 0.8
        self.place_at_grid(value_group, "E4", scale_factor=0.8)
        
        # Animate "opening"
        self.play(
            keys[1].animate.set_color(COLOR_V).scale(1.1),
            FadeIn(value_group, shift=DOWN)
        )
        
        # Connect Query to Value through Key
        connection = Arrow(query_group.get_bottom(), keys[1].get_top(), color=WHITE, buff=0.1)
        self.play(Create(connection))
        
        self.wait(2)
