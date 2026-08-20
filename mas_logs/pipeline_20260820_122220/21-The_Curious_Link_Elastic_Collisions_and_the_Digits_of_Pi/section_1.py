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
            "Consider two blocks on a frictionless surface.",
            "Block A has mass m.",
            "Block B has mass one hundred to the n.",
            "They collide elastically with a wall.",
            "We count the total number of collisions."
        ]
        self.setup_layout("The Setup: The 1D Collision Paradox", lecture_lines)
        
        # --- Create Assets ---
        # Mouse-block (mass m) - Using SVG Asset
        mouse = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE, fill_opacity=0.6)
        mouse_label = Tex("m", color=WHITE)
        mouse_label.next_to(mouse, DOWN, buff=0.1)
        mouse_group = VGroup(mouse, mouse_label)
        
        # Elephant-block (mass M) - Using SVG Asset
        elephant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=GREEN, fill_opacity=0.6)
        elephant_label = Tex("M", color=WHITE)
        elephant_label.next_to(elephant, DOWN, buff=0.1)
        elephant_group = VGroup(elephant, elephant_label)
        
        # Wall - Using SVG Asset
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=GREY)
        
        # Velocities
        vel_arrow = Arrow(start=LEFT, end=RIGHT, color=RED).scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(mouse_group, "C2", scale_factor=0.5)
        self.place_at_grid(elephant_group, "C4", scale_factor=0.8)
        self.place_at_grid(wall, "C6", scale_factor=1.0)
        self.play(FadeIn(mouse_group), FadeIn(elephant_group), FadeIn(wall))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Indicate(mouse_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(Indicate(elephant_label))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.place_at_grid(vel_arrow, "E2", scale_factor=0.6)
        self.play(Create(vel_arrow), mouse_group.animate.shift(RIGHT * 1.5))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        collision_point = Dot(color=YELLOW).move_to(self.grid["D3"])
        self.play(Flash(collision_point, color=YELLOW))
        self.wait(2)
