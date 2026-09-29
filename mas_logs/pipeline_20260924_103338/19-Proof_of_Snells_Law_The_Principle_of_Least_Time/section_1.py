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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Lifeguards aim to reach the swimmer quickly.",
            "Running on sand is faster than swimming.",
            "They bend their path at the shoreline.",
            "This minimizes their total travel time.",
            "Light follows this same optimized path."
        ]
        self.setup_layout("Proof of Snell's Law: The Principle of Least Time", lecture_lines)
        
        # Assets
        beach_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beach.svg")
        lifeguard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lifeguard.svg")

        # === Animation for Lecture Line 1 ===
        # Create sandy beach interface
        shoreline_box = Rectangle(width=4, height=3, fill_opacity=0.3, fill_color="#F4A460", stroke_width=0)
        self.place_in_area(shoreline_box, 'C3', 'E5', scale_factor=0.8)
        self.place_at_grid(beach_icon, 'B5', scale_factor=0.5)

        start_pt = Dot(color=WHITE)
        self.place_at_grid(start_pt, 'A2')
        end_pt = Dot(color=WHITE)
        self.place_at_grid(end_pt, 'F5')
        
        start_label = Text("Start", font_size=20)
        end_label = Text("End", font_size=20)
        self.place_at_grid(start_label, 'A3', scale_factor=0.7)
        self.place_at_grid(end_label, 'F4', scale_factor=0.7)
        
        self.add(shoreline_box, beach_icon, start_pt, end_pt, start_label, end_label)
        self.lecture[0].set_color("#FFFFFF")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.place_at_grid(lifeguard_icon, 'C4', scale_factor=0.4)
        lifeguard_label = Text("Lifeguard", font_size=18, color="#FF0000")
        self.place_at_grid(lifeguard_label, 'C5', scale_factor=0.7)
        self.add(lifeguard_icon, lifeguard_label)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        # Optimized path bending at shoreline
        path = VMobject()
        path.set_points_smoothly([start_pt.get_center(), [0.5, 0.5, 0], end_pt.get_center()])
        path.set_color("#FFFF00")
        self.place_in_area(path, 'A4', 'F6', scale_factor=0.9)
        path_label = Text("Path", font_size=18, color="#FFFF00")
        self.place_at_grid(path_label, 'B4', scale_factor=0.7)
        
        self.play(Create(path), Write(path_label))
        self.wait(2)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.wait(2)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        self.wait(2)
