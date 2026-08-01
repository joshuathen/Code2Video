from manim import *
import numpy as np
from pathlib import Path

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
        # Section Title and Lecture Lines
        title_text = "The Hook: The Thief's Dilemma"
        lecture_lines = [
            "Two thieves stole a necklace of rubies and emeralds.",
            "They must share the beads perfectly in half.",
            "What is the minimum number of cuts required?"
        ]
        self.setup_layout(title_text, lecture_lines)

        # Asset paths
        ruby_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/ruby.svg"
        emerald_asset = "/mmfs1/data/home/jthen/Code2Video/assets/icon/emerald.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line to signal current focus
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Beads: 6 Rubies, 4 Emeralds in a fixed sequence for visualization
        bead_types = ["R", "E", "R", "R", "E", "R", "E", "R", "E", "R"]
        beads = VGroup()
        for b_type in bead_types:
            asset_path = ruby_asset if b_type == "R" else emerald_asset
            # Load SVG assets and scale appropriately
            bead = SVGMobject(asset_path).scale(0.18)
            beads.add(bead)
        
        # Arrange beads horizontally
        beads.arrange(RIGHT, buff=0.35)
        
        # Create a horizontal line (the necklace) that connects the beads
        necklace_line = Line(beads.get_left(), beads.get_right(), color=WHITE)
        necklace_group = VGroup(necklace_line, beads)
        
        # Use grid system to place the necklace in row B (top-ish right side)
        self.place_in_area(necklace_group, "B1", "B6")
        
        self.play(Create(necklace_line), FadeIn(beads, shift=UP*0.1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture line highlighting
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Labels for 'Thief A' and 'Thief B' placed in the lower grid area
        thief_a = Text("Thief A", font_size=24, color=WHITE)
        thief_b = Text("Thief B", font_size=24, color=WHITE)
        self.place_at_grid(thief_a, "D2")
        self.place_at_grid(thief_b, "D5")
        
        # Display the goal target text with assets in the center (row C)
        goal_text = Text("Goal: 3 ", font_size=20, color=WHITE)
        goal_ruby = SVGMobject(ruby_asset).scale(0.15)
        goal_comma = Text(", 2 ", font_size=20, color=WHITE)
        goal_emerald = SVGMobject(emerald_asset).scale(0.15)
        
        goal_elements = VGroup(goal_text, goal_ruby, goal_comma, goal_emerald).arrange(RIGHT, buff=0.1)
        # Apply yellow highlight box as requested
        goal_box = SurroundingRectangle(goal_elements, color="#FFFF00", fill_opacity=0.3, stroke_width=2)
        goal_display = VGroup(goal_box, goal_elements)
        self.place_in_area(goal_display, "C1", "C6")
        
        self.play(
            Write(thief_a), 
            Write(thief_b), 
            FadeIn(goal_display)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture line highlighting
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Create a dashed vertical line to represent a cut point
        cut_line = DashedLine(UP*0.4, DOWN*0.4, color=WHITE)
        # Start at the left side of the necklace (grid position B1)
        self.place_at_grid(cut_line, "B1")
        
        self.play(Create(cut_line))
        # Animate the cut line moving along the necklace to B6
        self.play(
            cut_line.animate.move_to(self.grid["B6"]), 
            run_time=3, 
            rate_func=linear
        )
        self.play(FadeOut(cut_line))
        
        # Wrap up section
        self.wait(1)
        self.play(self.lecture[2].animate.set_color(WHITE))
