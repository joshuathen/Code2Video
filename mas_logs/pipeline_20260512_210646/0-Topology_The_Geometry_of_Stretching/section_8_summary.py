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

class Section8SummaryScene(TeachingScene):
    def construct(self):
        # Setup the scene layout
        lecture_lines_text = [
            'Topology finds the essence of shape beyond measurements.',
            'It categorizes the universe through connectivity and holes.',
            'This "rubber geometry" reveals the core of mathematical structure.'
        ]
        self.setup_layout("Summary: The Essence of Shape", lecture_lines_text)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line (Yellow)
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Create shapes: sphere (Circle), cube (Square), and donut (SVG Asset)
        # Note: Using yellow-ish colors for Line 1 items to match highlight
        sphere = Circle(radius=0.5, color="#FFFF00", fill_opacity=0.6)
        cube = Square(side_length=1.0, color="#FFFF00", fill_opacity=0.6)
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/donut.svg
        donut = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/donut.svg")
        donut.set_color("#FFFF00")
        
        # Position them according to layout issues 54 and 55
        self.place_at_grid(sphere, 'A2', scale_factor=0.8)
        self.place_at_grid(cube, 'A5', scale_factor=0.7)
        self.place_at_grid(donut, 'F4', scale_factor=0.8)
        
        self.play(FadeIn(sphere), FadeIn(cube), FadeIn(donut))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line (Cyan)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        # Create target shapes: points and circle (#FFFFFF per description)
        point1 = Dot(color="#FFFFFF", radius=0.1)
        point2 = Dot(color="#FFFFFF", radius=0.1)
        target_circle = Circle(radius=0.4, color="#FFFFFF", stroke_width=4)
        
        # Position target shapes to align with sources (Issues 54, 55)
        self.place_at_grid(point1, 'A2')
        self.place_at_grid(point2, 'A5')
        self.place_at_grid(target_circle, 'F4')
        
        # Morph animations
        self.play(
            ReplacementTransform(sphere, point1),
            ReplacementTransform(cube, point2),
            ReplacementTransform(donut, target_circle)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line (Green)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00")
        )
        
        # Display 'The Science of Continuity' in green
        summary_text = Text("The Science of Continuity", color="#00FF00", font_size=32)
        # Place in larger area with scale 1.0 (Issue 56)
        self.place_in_area(summary_text, 'C1', 'D6', scale_factor=1.0)
        
        self.play(
            FadeOut(point1),
            FadeOut(point2),
            FadeOut(target_circle),
            Write(summary_text)
        )
        self.wait(3)
