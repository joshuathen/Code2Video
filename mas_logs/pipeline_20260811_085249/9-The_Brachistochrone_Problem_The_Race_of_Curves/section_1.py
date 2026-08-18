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
        self.setup_layout("The Brachistochrone Problem", ["A bead slides under gravity from A to B.", "Which path takes the shortest time?", "Compare the straight line against a curve."])
        
        # Define assets
        bead_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg", color="#E74C3C")
        bead_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg", color="#E74C3C")
        label_a = Text("A", color="#E74C3C", font_size=24)
        label_b = Text("B", color="#E74C3C", font_size=24)
        
        # Positions based on criticisms
        self.place_at_grid(bead_a, 'B4', scale_factor=0.8)
        self.place_at_grid(bead_b, 'E4', scale_factor=0.8)
        label_a.next_to(bead_a, UP, buff=0.1)
        label_b.next_to(bead_b, DOWN, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        title_text = Text("The Brachistochrone Problem", color=WHITE, font_size=32).move_to(self.grid['A3'])
        self.play(Write(title_text))
        self.play(FadeIn(bead_a), FadeIn(bead_b), FadeIn(label_a), FadeIn(label_b))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        straight_path = Line(bead_a.get_center(), bead_b.get_center(), color=WHITE)
        self.play(Create(straight_path))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        # Creating a curve using a CubicBezier
        curve = CubicBezier(bead_a.get_center(), self.grid['C4'], self.grid['D4'], bead_b.get_center(), color="#3498DB")
        # Apply positioning fix for curve
        self.place_in_area(curve, 'B4', 'E4', scale_factor=0.7)
        self.play(Create(curve))
        
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg", color="#F1C40F")
        ball.scale(0.5).move_to(bead_a.get_center())
        self.play(MoveAlongPath(ball, curve), run_time=2)
        
        dashed_line = DashedLine(bead_a.get_center(), bead_b.get_center(), color="#2ECC71")
        self.play(Create(dashed_line))
        self.wait(1)
