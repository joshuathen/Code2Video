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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Boundary of Wonder: The Julia Set", [
            "The Julia Set marks the boundary of chaos.",
            "Here, tiny changes lead to drastically different futures.",
            "This complex edge is jagged and infinitely detailed."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw a white (#FFFFFF) jagged line representing a mountain ridge.
        self.lecture[0].set_color(WHITE)
        
        ridge_points = [
            self.grid['C1'],
            self.grid['C2'] + UP * 0.5,
            self.grid['C3'] + UP * 1.0,  # Peak
            self.grid['C4'] + UP * 0.2,
            self.grid['C5'] - UP * 0.3,
            self.grid['C6']
        ]
        ridge = VMobject(color=WHITE)
        ridge.set_points_as_corners(ridge_points)
        
        self.play(Create(ridge))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place a green point (#00FF00) and a red point (#FF0000) on the ridge; they fall into separate valleys.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(BLUE_A)
        
        peak_pos = self.grid['C3'] + UP * 1.0
        green_point = Dot(point=peak_pos, color="#00FF00", radius=0.1)
        red_point = Dot(point=peak_pos, color="#FF0000", radius=0.1)
        
        # Define valleys paths
        blue_path_points = [
            peak_pos,
            self.grid['D2'],
            self.grid['E1']
        ]
        blue_path = VMobject().set_points_as_corners(blue_path_points).make_smooth()
        
        red_path_points = [
            peak_pos,
            self.grid['D4'],
            self.grid['E6']
        ]
        red_path = VMobject().set_points_as_corners(red_path_points).make_smooth()
        
        blue_label = Text("Valley A", font_size=16, color=BLUE)
        self.place_at_grid(blue_label, 'F2', scale_factor=0.8) # Issue 32 fix
        
        red_label = Text("Valley B", font_size=16, color=RED)
        self.place_at_grid(red_label, 'F5', scale_factor=0.8) # Issue 33 fix

        self.play(FadeIn(green_point), FadeIn(red_point))
        self.play(FadeIn(blue_label), FadeIn(red_label))
        
        self.play(
            MoveAlongPath(green_point, blue_path),
            MoveAlongPath(red_point, red_path),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transform the ridge into a glowing fractal Julia set boundary (#FFFFFF).
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(WHITE)
        
        # Simple fractal generation by midpoint displacement
        def get_fractal_pts(pts, depth=3):
            curr = pts
            for _ in range(depth):
                next_pts = []
                for i in range(len(curr) - 1):
                    p1 = curr[i]
                    p2 = curr[i + 1]
                    mid = (p1 + p2) / 2
                    # Normal vector for displacement
                    direction = p2 - p1
                    normal = np.array([-direction[1], direction[0], 0])
                    if np.linalg.norm(normal) > 0:
                        normal = normal / np.linalg.norm(normal)
                    
                    # Alternating displacement
                    disp_mag = 0.3 * (0.5 ** _)
                    disp = normal * disp_mag * (1 if i % 2 == 0 else -1)
                    
                    next_pts.extend([p1, mid + disp])
                next_pts.append(curr[-1])
                curr = next_pts
            return curr

        f_pts = get_fractal_pts(ridge_points, depth=3)
        fractal_ridge = VMobject(color=WHITE).set_points_as_corners(f_pts)
        fractal_ridge.set_stroke(width=1.5)
        
        glow = fractal_ridge.copy().set_stroke(width=5, opacity=0.3)
        
        julia_label = Text("Julia Set", font_size=24, color=WHITE)
        self.place_at_grid(julia_label, 'A4', scale_factor=0.8) # Issue 31 fix

        self.play(
            Transform(ridge, fractal_ridge),
            FadeOut(green_point), FadeOut(red_point),
            FadeOut(blue_label), FadeOut(red_label),
            FadeIn(glow),
            Write(julia_label)
        )
        
        # Flash effect using a large rectangle
        flash = Rectangle(
            width=20, 
            height=20, 
            fill_color=WHITE, 
            fill_opacity=0.4, 
            stroke_width=0
        )
        self.play(FadeIn(flash), run_time=0.1)
        self.play(FadeOut(flash), run_time=0.4)
        
        self.wait(2)
