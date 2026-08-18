from manim import *

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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "Scaling to 2D: Image Processing"
        lines = [
            "In 2D, we process images as large grids of pixels.",
            "A smaller grid, called a filter, is placed on top.",
            "We highlight the specific region where the filter currently sits.",
            "The filter slides pixel by pixel across the entire image.",
            "Each step produces a new value in our feature map."
        ]
        
        self.setup_layout(title_text, lines)
        
        # === Animation for Lecture Line 1 ===
        # Display a 5x5 grid representing an image (#888888) using [Asset: image.svg]
        line1_color = "#888888"
        
        # Background Asset (Issue 18/36)
        image_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/image.svg")
        image_svg.set_color(line1_color).set_opacity(0.15)
        self.place_in_area(image_svg, "B4", "E5", scale_factor=0.8)
        
        # 5x5 Grid
        image_grid = VGroup(*[
            Square(side_length=0.45, stroke_width=2, color=line1_color)
            for _ in range(25)
        ]).arrange_in_grid(rows=5, cols=5, buff=0.05)
        
        # Align grid with corrected area (Issue 28/36: 'B4'-'E5', scale 0.8)
        self.place_in_area(image_grid, "B4", "E5", scale_factor=0.8)
        
        self.play(self.lecture[0].animate.set_color(line1_color))
        self.play(FadeIn(image_svg), Create(image_grid))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Display a 3x3 filter grid (#FFFF00) using [Asset: processing.svg]
        line2_color = "#FFFF00"
        
        # Filter Asset (Issue 18/36)
        filter_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/processing.svg")
        filter_svg.set_color(line2_color)
        
        # 3x3 Filter Squares
        filter_grid_squares = VGroup(*[
            Square(side_length=0.45, stroke_width=4, color=line2_color, fill_opacity=0.3, fill_color=line2_color)
            for _ in range(9)
        ]).arrange_in_grid(rows=3, cols=3, buff=0.05).scale(0.8)
        
        filter_group = VGroup(filter_svg.scale(0.3), filter_grid_squares)
        
        # Position filter at the top-left of the image grid (Center of the 3x3 region)
        filter_group.move_to(image_grid[6])
        
        self.play(self.lecture[1].animate.set_color(line2_color))
        self.play(FadeIn(filter_group))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight a 3x3 region on the image grid (#FFFFFF)
        line3_color = "#FFFFFF"
        
        def get_highlight_region(center_idx):
            row, col = divmod(center_idx, 5)
            indices = []
            for r in range(row-1, row+2):
                for c in range(col-1, col+2):
                    indices.append(r * 5 + c)
            return VGroup(*[image_grid[i].copy().set_stroke(line3_color, width=4) for i in indices])

        highlight = get_highlight_region(6)
        
        self.play(self.lecture[2].animate.set_color(line3_color))
        self.play(Create(highlight))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Animate the 3x3 filter sliding pixel by pixel (#00FFFF)
        line4_color = "#00FFFF"
        sliding_path = [6, 7, 8, 11, 12, 13, 16, 17, 18]
        
        self.play(self.lecture[3].animate.set_color(line4_color))
        self.play(
            filter_group.animate.set_color(line4_color),
            highlight.animate.set_color(line4_color)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Fill a new 3x3 output grid with computed feature values (#00FF00).
        line5_color = "#00FF00"
        
        # Output Grid (Issue 29/36: 'B6'-'D6', scale 0.7)
        output_grid = VGroup(*[
            Square(side_length=0.45, stroke_width=2, color=line5_color)
            for _ in range(9)
        ]).arrange_in_grid(rows=3, cols=3, buff=0.1)
        
        self.place_in_area(output_grid, "B6", "D6", scale_factor=0.7)
        
        # Output cell visual feedback
        output_fills = VGroup(*[
            Square(side_length=0.3, stroke_width=0, fill_opacity=0.8, fill_color=line5_color)
            .move_to(output_grid[i])
            for i in range(9)
        ])
        
        self.play(self.lecture[4].animate.set_color(line5_color))
        self.play(Create(output_grid))

        # Perform the sliding and filling
        for i, idx in enumerate(sliding_path):
            if i > 0:
                self.play(
                    filter_group.animate.move_to(image_grid[idx]),
                    highlight.animate.become(get_highlight_region(idx)),
                    run_time=0.3
                )
            
            # Produce output value
            self.play(FadeIn(output_fills[i], scale=0.5), run_time=0.2)
            
        self.wait(2)
