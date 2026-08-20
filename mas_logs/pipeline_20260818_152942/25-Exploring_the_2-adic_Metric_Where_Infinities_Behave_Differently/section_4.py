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
        self.setup_layout("Visualizing the 2-adic Ultrametric Property", [
            "Ultrametric spaces follow a stronger rule.",
            "Points land in nested bucket structures.",
            "Distance doesn't add up like paths."
        ])
        
        # Asset definition
        bucket_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bucket.svg"
        
        # --- Animation for Lecture Line 1 ---
        # Show three points forming a triangle using buckets
        b1 = SVGMobject(bucket_asset, color=WHITE)
        b2 = SVGMobject(bucket_asset, color=WHITE)
        b3 = SVGMobject(bucket_asset, color=WHITE)
        
        # Applying requested grid shifts: Column C, D, E instead of B, C, D
        self.place_at_grid(b1, 'C3', scale_factor=0.6)
        self.place_at_grid(b2, 'D4', scale_factor=0.6)
        self.place_at_grid(b3, 'E5', scale_factor=0.6)
        
        triangle = Polygon(b1.get_center(), b2.get_center(), b3.get_center(), color=WHITE)
        
        self.play(FadeIn(triangle, b1, b2, b3))
        self.lecture[0].set_color("#FFFFFF")

        # --- Animation for Lecture Line 2 ---
        # Visualize the strong triangle inequality using nested buckets
        c1 = SVGMobject(bucket_asset, color="#00FF00")
        c2 = SVGMobject(bucket_asset, color="#00FF00")
        
        self.place_at_grid(c1, 'D4', scale_factor=0.5)
        self.place_at_grid(c2, 'E5', scale_factor=0.5)
        
        self.play(FadeIn(c1, c2))
        self.lecture[1].set_color("#00FF00")

        # --- Animation for Lecture Line 3 ---
        # Demonstrate the ultrametric property visually.
        # Bouncing ball
        ball = Dot(color="#FFFF00", radius=0.15)
        self.place_at_grid(ball, 'C3', scale_factor=0.6)
        
        path = VMobject(color="#FFFF00")
        path.set_points_smoothly([self.grid['C3'], self.grid['D4'], self.grid['E5']])
        
        self.play(MoveAlongPath(ball, path), run_time=2)
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)
