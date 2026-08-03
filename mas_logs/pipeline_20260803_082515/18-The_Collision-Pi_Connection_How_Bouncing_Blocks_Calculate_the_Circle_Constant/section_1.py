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
        # Title and Lecture Lines
        title = "The Setup: A Simple Counting Game"
        lines = [
            "Meet our setup: a frictionless floor and a wall.",
            "We have a small block and a massive block.",
            "The larger block slides toward the smaller one.",
            "They bounce between each other and the wall.",
            "Our goal: count every single collision that occurs."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_WALL = "#808080"
        COLOR_FLOOR = "#FFFFFF"
        COLOR_MOUSE = "#00FF00"
        COLOR_ELEPHANT = "#FFA500"
        COLOR_COLLISION = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Floor: from F1 to F6 center
        floor = Line(self.grid["F1"] + LEFT*0.5, self.grid["F6"] + RIGHT*0.5, color=COLOR_FLOOR)
        # Wall: from B1 to F1
        wall = Line(self.grid["B1"] + UP*0.5, self.grid["F1"] + DOWN*0.5, color=COLOR_WALL, stroke_width=8)
        
        self.play(Create(floor), Create(wall), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Mouse block (smaller) - Fix Issue 26: move to E3
        mouse_block = Square(side_length=0.6, fill_opacity=1, color=COLOR_MOUSE)
        self.place_at_grid(mouse_block, "E3")
        mouse_label = Text("m", font_size=18, color=COLOR_MOUSE).next_to(mouse_block, UP, buff=0.1)
        
        # Elephant block (larger) - Fix Issue 27: move to E6
        elephant_block = Square(side_length=1.2, fill_opacity=1, color=COLOR_ELEPHANT)
        self.place_at_grid(elephant_block, "E6")
        elephant_label = Text("M", font_size=22, color=COLOR_ELEPHANT).next_to(elephant_block, UP, buff=0.1)
        
        self.play(FadeIn(mouse_block), FadeIn(mouse_label), FadeIn(elephant_block), FadeIn(elephant_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Elephant starts moving left
        v_tracker_elephant = ValueTracker(self.grid["E6"][0])
        elephant_block.add_updater(lambda m: m.set_x(v_tracker_elephant.get_value()))
        elephant_label.add_updater(lambda m: m.set_x(v_tracker_elephant.get_value()))
        
        # Collision calculation:
        # Mouse is at E3: x = 2.5. Half width = 0.3.
        # Elephant Half width = 0.6.
        # Collision at elephant_x - 0.6 = 2.5 + 0.3 => elephant_x = 3.4
        self.play(v_tracker_elephant.animate.set_value(3.4), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Function to show collision flash without blocking for long
        def flash_collision(pos):
            flash = Circle(radius=0.1, color=COLOR_COLLISION, fill_opacity=1).move_to(pos)
            self.add(flash)
            self.play(flash.animate.scale(3).set_opacity(0), run_time=0.2)
            self.remove(flash)

        # Position helpers for Mouse
        mouse_x_tracker = ValueTracker(self.grid["E3"][0])
        mouse_block.add_updater(lambda m: m.set_x(mouse_x_tracker.get_value()))
        mouse_label.add_updater(lambda m: m.set_x(mouse_x_tracker.get_value()))
        
        # Flash at first E-M collision (at edge of blocks)
        flash_collision(np.array([2.8, self.grid["E3"][1], 0])) 
        
        # Bounce 1: Mouse hits wall
        # Wall is at x=0.5. Collision x = 0.5 + 0.3 (radius) = 0.8
        self.play(
            mouse_x_tracker.animate.set_value(0.8),
            v_tracker_elephant.animate.set_value(3.3), # Slow movement of elephant
            run_time=0.8, rate_func=linear
        )
        flash_collision(np.array([0.5, self.grid["E3"][1], 0])) # Wall hit
        
        # Bounce 2: Mouse hits Elephant again
        # Elephant is at 3.3. Mouse moves to hit it.
        # Meet at elephant_x = 3.1, mouse_x = 3.1 - 0.9 = 2.2
        self.play(
            mouse_x_tracker.animate.set_value(2.2),
            v_tracker_elephant.animate.set_value(3.1),
            run_time=0.6, rate_func=linear
        )
        flash_collision(np.array([2.5, self.grid["E3"][1], 0])) # Edge collision roughly at 2.5
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Show goal highlight
        goal_rect = SurroundingRectangle(self.lecture[4], color=YELLOW, buff=0.1)
        self.play(Create(goal_rect))
        
        # Final slow movement back to wall
        self.play(
            mouse_x_tracker.animate.set_value(0.8),
            v_tracker_elephant.animate.set_value(3.0),
            run_time=1
        )
        flash_collision(np.array([0.5, self.grid["E3"][1], 0]))
        
        self.play(FadeOut(goal_rect))
        self.wait(2)

        # Cleanup updaters
        mouse_block.clear_updaters()
        elephant_block.clear_updaters()
        mouse_label.clear_updaters()
        elephant_label.clear_updaters()
