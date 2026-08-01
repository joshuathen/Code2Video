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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the scene
        lines = [
            "Topology shifts our focus from measurements to structural connections.",
            "We identify deep patterns in DNA, data, and galaxies.",
            "In this flexible world, continuity is the ultimate rule."
        ]
        self.setup_layout("Summary: The Big Picture", lines)
        
        # Define Colors
        GENUS0_COLOR = "#56B4E9"
        GENUS1_COLOR = "#009E73"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Buckets
        bucket0 = RoundedRectangle(corner_radius=0.2, height=1.5, width=2.2, color=GENUS0_COLOR)
        bucket0_label = Text("Genus 0", font_size=20, color=GENUS0_COLOR)
        bucket0_group = VGroup(bucket0, bucket0_label.next_to(bucket0, UP, buff=0.2))
        self.place_in_area(bucket0_group, 'D1', 'F3')
        
        bucket1 = RoundedRectangle(corner_radius=0.2, height=1.5, width=2.2, color=GENUS1_COLOR)
        bucket1_label = Text("Genus 1", font_size=20, color=GENUS1_COLOR)
        bucket1_group = VGroup(bucket1, bucket1_label.next_to(bucket1, UP, buff=0.2))
        self.place_in_area(bucket1_group, 'D4', 'F6')
        
        self.play(FadeIn(bucket0_group), FadeIn(bucket1_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GENUS0_COLOR))
        
        # Objects for Genus 0
        sphere = Circle(radius=0.3, color=GENUS0_COLOR, fill_opacity=0.6)
        cube = Square(side_length=0.5, color=GENUS0_COLOR, fill_opacity=0.6)
        
        # Fixed layout and scaling based on VideoCritic feedback
        self.place_at_grid(sphere, 'B1', scale_factor=1.2)
        self.place_at_grid(cube, 'B3', scale_factor=1.2)
        
        self.play(FadeIn(sphere), FadeIn(cube))
        self.wait(0.5)
        
        # Move objects into bucket 0
        target0 = bucket0.get_center()
        self.play(
            sphere.animate.move_to(target0).scale(0.5),
            cube.animate.move_to(target0).scale(0.5),
            run_time=2
        )
        self.play(FadeOut(sphere), FadeOut(cube)) # Objects go "into" the bucket
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GENUS1_COLOR))
        
        # Objects for Genus 1
        # Create a simple Torus (Annulus)
        torus = Annulus(inner_radius=0.15, outer_radius=0.3, color=GENUS1_COLOR, fill_opacity=0.6)
        
        # Create a simple Mug
        mug_body = RoundedRectangle(corner_radius=0.1, height=0.5, width=0.4, color=GENUS1_COLOR, fill_opacity=0.6)
        mug_handle = Arc(radius=0.15, start_angle=-PI/2, angle=PI, color=GENUS1_COLOR).next_to(mug_body, RIGHT, buff=-0.1)
        mug = VGroup(mug_body, mug_handle)
        
        # Fixed layout and scaling based on VideoCritic feedback
        self.place_at_grid(mug, 'B4', scale_factor=1.2)
        self.place_at_grid(torus, 'B6', scale_factor=1.2)
        
        self.play(FadeIn(mug), FadeIn(torus))
        self.wait(0.5)
        
        # Move objects into bucket 1
        target1 = bucket1.get_center()
        self.play(
            mug.animate.move_to(target1).scale(0.5),
            torus.animate.move_to(target1).scale(0.5),
            run_time=2
        )
        self.play(FadeOut(mug), FadeOut(torus)) # Objects go "into" the bucket
        self.wait(2)
