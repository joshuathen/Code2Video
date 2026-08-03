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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Prerequisite Map: The Three Pillars", 
            [
                "Every power relationship involves a base and an exponent.",
                "Multiplying the base repeatedly gives us the final result.",
                "These three values form the foundation of our triangle."
            ]
        )

        # Colors
        COLOR_BASE = "#0000FF"    # Blue
        COLOR_EXPONENT = "#00FF00" # Green
        COLOR_RESULT = "#FF0000"   # Red

        # Asset path
        CUBE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg"

        def create_cube_asset(color):
            # Load asset once and style it
            cube = SVGMobject(CUBE_ASSET)
            cube.set_color(color)
            cube.set_stroke(width=1, color=WHITE)
            return cube

        # === Animation for Lecture Line 1 ===
        # Display 'Base: 2' in blue (#0000FF) alongside a single 3D-looking cube [Asset: .../cube.svg]
        self.play(self.lecture[0].animate.set_color(COLOR_BASE))
        
        base_label = Text("Base: 2", font_size=24, color=COLOR_BASE)
        self.place_at_grid(base_label, "B2")
        
        single_cube = create_cube_asset(COLOR_BASE)
        self.place_at_grid(single_cube, "B4", scale_factor=1.2)
        
        self.play(
            Write(base_label),
            FadeIn(single_cube)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display 'Exponent: 3' in green (#00FF00) as the cube [Asset: .../cube.svg] multiplies into a 2x2x2 block.
        self.play(self.lecture[1].animate.set_color(COLOR_EXPONENT))
        
        exponent_label = Text("Exponent: 3", font_size=24, color=COLOR_EXPONENT)
        # Fix from Issue 38: Update code around line 102 to: self.place_at_grid(exponent_label, 'D2')
        self.place_at_grid(exponent_label, 'D2')

        # Create a 2x2x2 block of cubes using the asset
        block_cubes = VGroup()
        side_spacing = 0.4
        depth_offset = 0.15
        
        # Layering from back to front (z: 0 to 1)
        for z in range(2): 
            for y in range(2): 
                for x in range(2):
                    c = create_cube_asset(COLOR_EXPONENT)
                    # Simple pseudo-3D stack
                    pos = np.array([
                        x * side_spacing + (1-z) * depth_offset, 
                        -y * side_spacing + (1-z) * depth_offset, 
                        0
                    ])
                    c.move_to(pos)
                    block_cubes.add(c)
        
        # Fix from Issue 37: Update code around line 121 to: self.place_in_area(block_cubes, 'D4', 'F5', scale_factor=0.8)
        self.place_in_area(block_cubes, 'D4', 'F5', scale_factor=0.8)
        
        self.play(
            Write(exponent_label),
            ReplacementTransform(single_cube, block_cubes)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display 'Result: 8' in red (#FF0000) as a label for the completed 8-cube block made of multiple cubes [Asset: .../cube.svg].
        self.play(self.lecture[2].animate.set_color(COLOR_RESULT))
        
        result_label = Text("Result: 8", font_size=24, color=COLOR_RESULT)
        # Fix from Issue 39: Update code around line 135 to: self.place_at_grid(result_label, 'E2')
        self.place_at_grid(result_label, 'E2')
        
        # Highlight the block in red
        highlight_rect = SurroundingRectangle(block_cubes, color=COLOR_RESULT, buff=0.1)
        
        self.play(
            Write(result_label),
            Create(highlight_rect),
            block_cubes.animate.set_color(COLOR_RESULT)
        )
        self.wait(2)
