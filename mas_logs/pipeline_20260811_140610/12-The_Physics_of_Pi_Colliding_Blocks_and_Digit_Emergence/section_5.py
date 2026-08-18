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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Collisions are not magic, just physics.",
            "Conservation laws govern every complex interaction.",
            "Mathematics describes the physical world perfectly."
        ]
        self.setup_layout("Synthesis & Summary", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display block collision summary screen.
        rect = RoundedRectangle(corner_radius=0.1, height=2, width=3, color=WHITE)
        label = Text("Collision Summary", font_size=24).next_to(rect, UP)
        group1 = VGroup(rect, label)
        self.place_in_area(group1, 'B4', 'D5', scale_factor=0.8)
        self.play(FadeIn(group1))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Show conservation law icons orbiting.
        icons = VGroup(*[Circle(radius=0.3, color="#00FFFF") for _ in range(2)])
        icons.arrange(RIGHT, buff=0.5)
        self.place_at_grid(icons, 'E4', scale_factor=0.8)
        self.play(FadeIn(icons))
        self.play(Rotating(icons, radians=2*PI, run_time=2))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Final animation: Math and physics merging.
        math_symbol = MathTex(r"\pi", font_size=48, color="#FFD700")
        phys_symbol = Text("F=ma", font_size=24, color="#FFD700")
        merge = VGroup(math_symbol, phys_symbol).arrange(DOWN)
        self.place_at_grid(merge, 'B5', scale_factor=1.2)
        self.play(GrowFromCenter(merge))
        self.lecture[2].set_color("#FFD700")
        
        self.wait(2)
