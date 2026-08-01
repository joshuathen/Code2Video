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
        # Configuration
        lines = [
            'Meet Leo, our lion residing on a 2D grid.',
            "Matrices can transform space and change Leo's shape.",
            "Leo's area scales according to the matrix used.",
            'This scaling factor is known as the determinant.',
            'A determinant of six means six times the area.'
        ]
        self.setup_layout("The Hook: Meet Leo the Grid-Lion", lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Define a coordinate system for the right side (visual anchor)
        # We'll use a local coordinate system based on the C3 area
        right_center = self.grid["D4"] # roughly middle of right side
        
        # Create a grid manually to avoid huge objects
        grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": "#D3D3D3",
                "stroke_width": 1,
                "stroke_opacity": 0.5
            },
            axis_config={"include_numbers": False, "stroke_opacity": 0}
        )
        self.place_at_grid(grid, "D4")
        
        # Asset: Leo the Lion
        # Load and place at origin of the grid
        leo = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg")
        leo.set_stroke(WHITE, width=1)
        leo.set_fill(WHITE, opacity=0.8)
        
        # Scale Leo to fit a 1x1 unit square
        leo.set_height(0.9)
        leo.move_to(grid.coords_to_point(0.5, 0.5)) # Place in the unit square [0,1]x[0,1]

        self.play(Create(grid), DrawBorderThenFill(leo))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Matrix for det=6: [[3, 0], [0, 2]] (Scaling x by 3, y by 2)
        matrix = [[3, 0], [0, 2]]
        
        # We transform the whole coordinate space (Leo + Grid)
        self.play(
            grid.animate.apply_matrix(matrix),
            leo.animate.apply_matrix(matrix),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Blue area (Original unit square)
        # Note: In manim, once transformed, finding the "original" spot requires calculating inverse
        # Or just knowing it was [0,1]x[0,1] at origin of grid.
        # However, to overlay precisely, we create a polygon transformed by the same matrix
        
        # Original area polygon
        original_rect = Polygon(
            grid.coords_to_point(0,0), grid.coords_to_point(1,0), 
            grid.coords_to_point(1,1), grid.coords_to_point(0,1),
            fill_color="#0000FF", fill_opacity=0.4, stroke_width=0
        )
        
        # Transformed area polygon (the whole thing)
        # Since matrix is [[3,0],[0,2]], the unit square is now [0,3]x[0,2]
        transformed_rect = Polygon(
            grid.coords_to_point(0,0), grid.coords_to_point(1,0), 
            grid.coords_to_point(1,1), grid.coords_to_point(0,1),
            fill_color="#FFA500", fill_opacity=0.4, stroke_width=2, stroke_color=ORANGE
        )

        # To show "original vs new", we'll just show the new orange area first
        self.play(FadeIn(transformed_rect))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        det_label = Text("Determinant", font_size=24, color=WHITE)
        self.place_at_grid(det_label, "A4")
        
        arrow = Arrow(
            start=det_label.get_bottom(),
            end=transformed_rect.get_center(),
            color=WHITE,
            buff=0.1
        )
        
        self.play(Write(det_label), GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        scale_text = Text("Scale factor: 6", font_size=32, color="#FFFF00")
        self.place_at_grid(scale_text, "F4")
        
        # Highlight original vs transformed area ratio
        # Show a small 1x1 blue box for comparison
        # We need to manually calculate where [0,1]x[0,1] is in transformed space
        # But for simple visual, we just draw it over
        orig_box_indicator = Rectangle(width=1, height=1, color="#0000FF", fill_opacity=0.6)
        orig_box_indicator.move_to(grid.coords_to_point(0.166, 0.25)) # just small marker
        # Better: just use a scaled down version of the current rect or original
        
        self.play(Write(scale_text))
        
        # Flash the transformed area to emphasize '6'
        self.play(Indicate(transformed_rect, color=YELLOW, scale_factor=1.1))
        self.wait(2)
