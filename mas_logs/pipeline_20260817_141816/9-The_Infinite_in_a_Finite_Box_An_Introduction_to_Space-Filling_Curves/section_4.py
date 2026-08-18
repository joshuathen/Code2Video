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
        self.setup_layout("Real-World Application: The Hilbert Curve", [
            "Hilbert curves map 2D data to 1D.", 
            "They preserve spatial locality in memory.", 
            "Useful for image compression and processing."
        ])

        # Define Hilbert curve generator
        def get_hilbert_curve(order, scale=1.5):
            def d2xy(n, d):
                t = d
                x = 0
                y = 0
                s = 1
                while s < n:
                    rx = 1 & (t // 2)
                    ry = 1 & (t ^ rx)
                    if ry == 0:
                        if rx == 1:
                            x = s - 1 - x
                            y = s - 1 - y
                        x, y = y, x
                    x += s * rx
                    y += s * ry
                    t //= 4
                    s *= 2
                return x, y
            
            n = 2**order
            points = []
            for i in range(n * n):
                x, y = d2xy(n, i)
                points.append(np.array([(x - (n - 1) / 2) / n * scale, (y - (n - 1) / 2) / n * scale, 0]))
            return VMobject().set_points_as_corners(points)

        # 1. Show square area + camera icon
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        square = Square(side_length=2.5, color="#FFFFFF")
        
        # Place square in area and camera icon near it
        self.place_in_area(square, 'B4', 'E6', scale_factor=1.1)
        camera_icon.next_to(square, UP)
        
        self.play(Create(square), FadeIn(camera_icon))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00CCFF")
        curve1 = get_hilbert_curve(1).set_color("#00CCFF")
        self.place_in_area(curve1, 'B4', 'E6', scale_factor=1.0)
        self.play(Create(curve1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF33")
        curve2 = get_hilbert_curve(2).set_color("#00FF33")
        self.place_in_area(curve2, 'B4', 'E6', scale_factor=1.0)
        self.play(Transform(curve1, curve2))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFCC00")
        processor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/processor.svg")
        processor_icon.next_to(square, DOWN)
        
        curve3 = get_hilbert_curve(3).set_color("#FFCC00")
        self.place_in_area(curve3, 'B4', 'E6', scale_factor=1.0)
        
        self.play(Transform(curve1, curve3), FadeIn(processor_icon))
        
        self.wait(2)
