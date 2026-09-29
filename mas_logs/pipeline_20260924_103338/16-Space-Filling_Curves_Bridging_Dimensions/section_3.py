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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Hilbert Curve: Optimization", [
            "The Hilbert curve follows a U-shape.",
            "It preserves locality between adjacent points.",
            "This makes it efficient for spatial indexing."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Load asset
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        self.place_at_grid(map_icon, "A5", scale_factor=0.3)
        
        def get_hilbert_path(order=2):
            path = []
            def recursive_hilbert(x, y, xi, xj, yi, yj, n):
                if n <= 0:
                    path.append(np.array([x + (xi + yi)/2, y + (xj + yj)/2, 0]))
                else:
                    recursive_hilbert(x, y, yi/2, yj/2, xi/2, xj/2, n-1)
                    recursive_hilbert(x+xi/2, y+xj/2, xi/2, xj/2, yi/2, yj/2, n-1)
                    recursive_hilbert(x+xi/2+yi/2, y+xj/2+yj/2, xi/2, xj/2, yi/2, yj/2, n-1)
                    recursive_hilbert(x+xi/2+yi-1, y+xj/2+yj-1, -yi/2, -yj/2, -xi/2, -xj/2, n-1)
            recursive_hilbert(0, 0, 1, 0, 0, 1, order)
            return VMobject().set_points_as_corners(path)

        hilbert = get_hilbert_path(order=2)
        hilbert.set_color("#FFFFFF")
        self.place_in_area(hilbert, "B3", "E5", scale_factor=0.9)
        self.play(Create(hilbert), FadeIn(map_icon), self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Recursion step
        self.play(hilbert.animate.set_color("#FF00FF"), self.lecture[1].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        comp_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(comp_icon, "E2", scale_factor=0.3)
        
        dots = VGroup(*[Dot(color="#00FFFF", radius=0.08) for _ in range(2)])
        # Position dots near each other on the curve
        pts = hilbert.get_points()
        dots[0].move_to(pts[0])
        dots[1].move_to(pts[1])
        
        self.add(dots)
        self.play(FadeIn(comp_icon), self.lecture[2].animate.set_color("#00FFFF"))
        self.play(FadeIn(dots))
        self.wait(1)
