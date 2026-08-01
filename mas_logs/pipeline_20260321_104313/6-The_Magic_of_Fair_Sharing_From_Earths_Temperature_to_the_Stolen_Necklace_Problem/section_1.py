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
        # Initial layout setup
        title_text = "The Thief's Dilemma: The Stolen Necklace Problem"
        lecture_lines = [
            "Two thieves steal a necklace with different beads.",
            "They want to split each bead type equally.",
            "How many cuts are needed for a fair share?",
            "Imagine ten red and eight green beads.",
            "Can we solve this with just two cuts?"
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors as defined in requirements
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#00FF00"
        THIEF_A_COLOR = "#FFD700"
        THIEF_B_COLOR = "#ADFF2F"
        MARKER_COLOR = "#FFA500"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # "Two thieves steal a necklace with different beads."
        self.lecture[0].set_color(THIEF_A_COLOR)
        
        # Sequence logic: R,R,G | G,R,R | G,G,R | R,G,G | R,R,G | G,R,R
        # This provides a 10 Red, 8 Green split solvable with 2 cuts at indices 4 and 13.
        bead_colors = [
            RED_COLOR, RED_COLOR, GREEN_COLOR,
            GREEN_COLOR, RED_COLOR, RED_COLOR,
            GREEN_COLOR, GREEN_COLOR, RED_COLOR,
            RED_COLOR, GREEN_COLOR, GREEN_COLOR,
            RED_COLOR, RED_COLOR, GREEN_COLOR,
            GREEN_COLOR, RED_COLOR, RED_COLOR
        ]
        
        beads = VGroup()
        for color in bead_colors:
            beads.add(Dot(radius=0.15, color=color))
        
        # Use grid system for beads (Row B)
        # We group beads in triplets to fit across the 6 grid units (B1-B6)
        triplets = VGroup()
        for i in range(0, 18, 3):
            triplet = beads[i:i+3].arrange(RIGHT, buff=0.15)
            self.place_at_grid(triplet, f"B{i//3 + 1}")
            triplets.add(triplet)
            
        necklace_string = VGroup()
        for i in range(len(beads) - 1):
            line = Line(beads[i].get_center(), beads[i+1].get_center(), stroke_width=2, color=WHITE)
            necklace_string.add(line)
            
        self.play(Create(beads), Create(necklace_string))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "They want to split each bead type equally."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(THIEF_B_COLOR)
        
        # Highlight figures for thieves
        thief_a = Circle(radius=0.4, color=THIEF_A_COLOR, fill_opacity=0.5)
        self.place_at_grid(thief_a, 'D2')
        label_a = Text("Thief A", font_size=18, color=THIEF_A_COLOR)
        label_a.next_to(thief_a, DOWN, buff=0.2)
        
        thief_b = Circle(radius=0.4, color=THIEF_B_COLOR, fill_opacity=0.5)
        self.place_at_grid(thief_b, 'D5')
        label_b = Text("Thief B", font_size=18, color=THIEF_B_COLOR)
        label_b.next_to(thief_b, DOWN, buff=0.2)
        
        target_a = Text("Target: 5 Red, 4 Green", font_size=16, color=WHITE)
        target_a.next_to(label_a, DOWN, buff=0.2)
        target_b = target_a.copy()
        target_b.next_to(label_b, DOWN, buff=0.2)
        
        self.play(FadeIn(thief_a, thief_b, label_a, label_b, target_a, target_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "How many cuts are needed for a fair share?"
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Show many cuts (one between every pair of beads)
        many_cuts = VGroup()
        for i in range(len(beads) - 1):
            mid = (beads[i].get_center() + beads[i+1].get_center()) / 2
            cut = Line(mid + UP*0.3, mid + DOWN*0.3, color=WHITE, stroke_width=1)
            many_cuts.add(cut)
            
        self.play(Create(many_cuts))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Imagine ten red and eight green beads."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED_COLOR)
        
        # Pulse animation to focus on the bead count
        self.play(beads.animate.scale(1.15), run_time=0.4)
        self.play(beads.animate.scale(1/1.15), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Can we solve this with just two cuts?"
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(MARKER_COLOR)
        
        # Remove tedious cuts and show the two optimal orange cuts
        self.play(FadeOut(many_cuts))
        
        # Cut 1: After 4 beads (index 3|4)
        mid1 = (beads[3].get_center() + beads[4].get_center()) / 2
        cut1 = Line(mid1 + UP*0.6, mid1 + DOWN*0.6, color=MARKER_COLOR, stroke_width=5)
        
        # Cut 2: After 13 beads (index 12|13)
        mid2 = (beads[12].get_center() + beads[13].get_center()) / 2
        cut2 = Line(mid2 + UP*0.6, mid2 + DOWN*0.6, color=MARKER_COLOR, stroke_width=5)
        
        self.play(Create(cut1), Create(cut2))
        self.wait(1)
        
        # Group segments based on cuts
        # Thief B gets 0-3 and 13-17
        # Thief A gets 4-12
        seg1 = VGroup(*beads[0:4], *necklace_string[0:3])
        seg2 = VGroup(*beads[4:13], *necklace_string[4:12])
        seg3 = VGroup(*beads[13:18], *necklace_string[13:17])
        
        # Sever the string at the cut locations
        self.play(FadeOut(necklace_string[3]), FadeOut(necklace_string[12]))
        
        # Final distribution: move segments to thieves
        # Use grid positions for destination centers
        self.play(
            seg2.animate.move_to(self.grid['D2']).scale(0.5),
            VGroup(seg1, seg3).animate.move_to(self.grid['D5']).scale(0.5),
            FadeOut(cut1), FadeOut(cut2)
        )
        self.wait(2)
