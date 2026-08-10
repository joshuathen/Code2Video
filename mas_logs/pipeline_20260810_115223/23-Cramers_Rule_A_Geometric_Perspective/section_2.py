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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Linear systems represent vector combinations.",
            "Solve for weights x and y.",
            "Reach point b using v1, v2."
        ]
        self.setup_layout("Framing the Linear System", lecture_lines)
        
        # Setup vectors and drone
        v1 = Arrow(ORIGIN, [1.5, 0.75, 0], color=YELLOW)
        v2 = Arrow(ORIGIN, [0.75, 2.25, 0], color=YELLOW)
        vecs = VGroup(v1, v2)
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg", color=YELLOW)
        
        # Combined group for spatial consistency
        vis_group = VGroup(vecs, drone)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(vis_group, 'C4', 'F6', scale_factor=0.45)
        self.play(Create(vecs), FadeIn(drone))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Parallelogram visual
        p = Polygon(ORIGIN, v1.get_end(), v1.get_end() + v2.get_end(), v2.get_end(), color=BLUE, fill_opacity=0.3)
        self.play(Create(p))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Target point
        dot = Dot(v1.get_end() + v2.get_end(), color=RED)
        b_label = Text("b", font_size=24, color=RED)
        self.place_at_grid(b_label, 'E5', scale_factor=0.8)
        b_label.next_to(dot, UR, buff=0.1)
        
        self.play(FadeIn(dot), Write(b_label))
        self.wait(1)
