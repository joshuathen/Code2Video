from manim import *
import os

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
        self.setup_layout("Introduction: The Geometric Puzzle", [
            "Place points on a circle.", 
            "Connect every pair with chords.", 
            "How many regions are created?"
        ])
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg"
        if os.path.exists(asset_path):
            circle_icon = SVGMobject(asset_path, color="#3498DB")
        else:
            circle_icon = Circle(radius=1.0, color="#3498DB")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#E74C3C")
        self.place_at_grid(circle_icon, 'D4', scale_factor=0.9)
        self.play(FadeIn(circle_icon))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        
        # Positioning for line 2
        self.place_in_area(circle_icon, 'C3', 'E5', scale_factor=0.85)
        
        # Visualizing a polygon/chords - Use circle path if SVGMobject is not a path
        circle_path = Circle(radius=circle_icon.width / 2).move_to(circle_icon.get_center())
        points = [circle_path.point_from_proportion(p / 6) for p in range(6)]
        dots = VGroup(*[Dot(p, color="#E74C3C") for p in points])
        self.add(dots)
        chords = VGroup()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                chords.add(Line(points[i], points[j], color="#3498DB"))
        self.play(Create(chords), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#F1C40F")
        # Final positioning
        self.place_at_grid(circle_icon, 'E5', scale_factor=0.75)
        self.wait(1)
