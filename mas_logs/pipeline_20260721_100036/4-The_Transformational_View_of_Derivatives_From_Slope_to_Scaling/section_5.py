from manim import *
import numpy as np

CYAN = TEAL

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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Application: The Magnifying Glass Effect",
            [
                "This view helps computers distort textures for graphics.",
                "Digital magnifying glasses scale pixels using local derivatives.",
                "Derivatives ensure smooth and accurate visual scaling."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Initial state: Highlight line 1 in yellow
        self.lecture[0].set_color(YELLOW)
        
        # Texture grid: 5x5 collection of squares
        texture_grid = VGroup(*[
            Square(side_length=0.6, stroke_width=2, color=GRAY_B)
            for _ in range(25)
        ]).arrange_in_grid(rows=5, cols=5, buff=0.05)
        
        # Positioning texture_grid in the area C3 to F6 as per Issue 36/38
        self.place_in_area(texture_grid, "C3", "F6", scale_factor=0.8)
        
        # Magnifying glass: Using Asset as per Issue 23
        # Asset path: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg")
        magnifying_glass.set_color(WHITE)
        
        # Initial position of the glass at C3 as per Issue 37
        self.place_at_grid(magnifying_glass, "C3", scale_factor=0.6)
        
        # Animation: Fade in grid and create magnifying glass, then move across to F6
        self.play(FadeIn(texture_grid))
        self.play(FadeIn(magnifying_glass)) # SVGMobject usually better with FadeIn or DrawBorder
        self.play(magnifying_glass.animate.move_to(self.grid["F6"]), run_time=2.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Update lecture colors: move highlight to line 2 (Cyan)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(CYAN)
        
        # Target a specific cell (middle of the grid, index 12)
        target_cell = texture_grid[12]
        target_pos = target_cell.get_center()
        
        # Move glass over the target cell
        self.play(magnifying_glass.animate.move_to(target_pos), run_time=1.5)
        
        # Highlight and expand the cell
        self.play(
            target_cell.animate.set_fill("#00FFFF", opacity=0.6).set_stroke("#00FFFF", width=4),
        )
        self.play(
            target_cell.animate.scale(2.0),
            run_time=1.2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture colors: move highlight to line 3 (Green)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        
        # Derivative label follows the glass
        deriv_label = MathTex("f'(x) = 2.0", color=GREEN).scale(0.8)
        # Position relative to the magnifying glass SVG
        deriv_label.next_to(magnifying_glass, UP, buff=0.2)
        
        # Use updater to keep label attached to the glass during movement
        def label_follow_updater(m):
            m.next_to(magnifying_glass, UP, buff=0.2)
            
        deriv_label.add_updater(label_follow_updater)
        
        self.play(Write(deriv_label))
        
        # Demonstrate movement with derivative label following
        # Moving to some different grid points
        self.play(magnifying_glass.animate.move_to(self.grid["D4"]), run_time=1.5)
        self.play(magnifying_glass.animate.move_to(self.grid["E5"]), run_time=1.5)
        
        self.wait(2)
        
        # Clean up updater before end of scene
        deriv_label.remove_updater(label_follow_updater)
