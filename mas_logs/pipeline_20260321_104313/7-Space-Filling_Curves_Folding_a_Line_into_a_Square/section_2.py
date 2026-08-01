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
        # Initialize the lecture lines and title
        title_text = "Prerequisite: Iteration and Limits"
        lecture_lines = [
            "Space-filling curves result from repeating a simple process.",
            "We divide a square into four smaller quadrants.",
            "Infinite repetition creates a dense, nested grid structure."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Distinct colors for each stage to match lecture lines with animation elements
        COLOR_1 = "#FFFF00"  # Yellow
        COLOR_2 = "#00FFFF"  # Cyan
        COLOR_3 = "#00FF00"  # Green
        NORMAL_WHITE = "#FFFFFF"

        # Helper function for recursive subdivision grid
        def get_subdivision_grid(box, level, color, stroke_width=2):
            grid_lines = VGroup()
            ul = box.get_corner(UL)
            ur = box.get_corner(UR)
            dl = box.get_corner(DL)
            side = box.width
            
            num_divs = 2**level
            step = side / num_divs
            
            for i in range(1, num_divs):
                # Vertical lines
                v_start = ul + RIGHT * i * step
                v_end = dl + RIGHT * i * step
                grid_lines.add(Line(v_start, v_end, color=color, stroke_width=stroke_width))
                
                # Horizontal lines
                h_start = ul + DOWN * i * step
                h_end = ur + DOWN * i * step
                grid_lines.add(Line(h_start, h_end, color=color, stroke_width=stroke_width))
            return grid_lines

        # === Animation for Lecture Line 1 ===
        # Match Line 1 color with the square
        self.lecture[0].set_color(COLOR_1)
        
        # Create a large square centered in the right-side grid area (B2 to F6)
        # Shifted from A2-E6 to B2-F6 to avoid title occlusion and use bottom space.
        main_square = Square(side_length=4.0, color=COLOR_1)
        self.place_in_area(main_square, "B2", "F6")
        
        self.play(Create(main_square))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Match Line 2 color with the first level grid
        self.lecture[0].set_color(NORMAL_WHITE)
        self.lecture[1].set_color(COLOR_2)
        
        grid_level_1 = get_subdivision_grid(main_square, 1, color=COLOR_2, stroke_width=3)
        self.play(Create(grid_level_1))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Match Line 3 color with the dense subdivision process
        self.lecture[1].set_color(NORMAL_WHITE)
        self.lecture[2].set_color(COLOR_3)
        
        # Recursive subdivision: level 2, 3, 4, 5
        current_grid = grid_level_1
        for level in [2, 3, 4, 5]:
            # Decrease stroke width as density increases for clarity
            sw = max(0.5, 3.5 - level * 0.5)
            next_grid = get_subdivision_grid(main_square, level, color=COLOR_3, stroke_width=sw)
            
            # Change the square color to match the dense grid for consistency in the final stage
            if level == 2:
                self.play(
                    ReplacementTransform(current_grid, next_grid),
                    main_square.animate.set_color(COLOR_3),
                    run_time=1.0
                )
            else:
                self.play(ReplacementTransform(current_grid, next_grid), run_time=0.8)
            
            current_grid = next_grid
        
        self.wait(3)
