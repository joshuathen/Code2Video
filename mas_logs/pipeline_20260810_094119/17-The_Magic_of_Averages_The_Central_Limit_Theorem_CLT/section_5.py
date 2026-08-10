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
            "This allows inferences about massive populations.",
            "Essential for quality control in industry.",
            "Foundational for modern scientific polling."
        ]
        self.setup_layout("Real-World Application: Why It Matters", lecture_lines)
        
        # Define assets
        factory = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/factory.svg")
        battery_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/battery.svg")
        
        batteries = VGroup(*[battery_icon.copy().set_color("#F1C40F") for _ in range(30)])
        mean_box = Square(side_length=0.8, color=BLUE).add(Text("Mean", font_size=16))
        checkmark = Tex(r"$\checkmark$", color="#2ECC71").scale(1.0)

        # === Animation for Lecture Line 1 ===
        # Show factory line
        self.place_at_grid(factory, 'B2', scale_factor=0.5)
        self.place_in_area(batteries.arrange(RIGHT, buff=0.05), 'B3', 'B6', scale_factor=0.4)
        self.play(FadeIn(factory), FadeIn(batteries))
        self.lecture[0].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate to mean calculator
        self.place_at_grid(mean_box, 'C3', scale_factor=0.8)
        self.play(
            FadeIn(mean_box),
            *[battery.animate.move_to(mean_box.get_center()).scale(0.1) for battery in batteries]
        )
        self.lecture[1].set_color("#F1C40F")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash checkmark
        self.place_at_grid(checkmark, 'C4', scale_factor=1.0)
        self.play(Flash(checkmark.get_center(), color="#2ECC71", line_length=0.2, num_lines=12))
        self.add(checkmark)
        self.lecture[2].set_color("#E74C3C")
        self.wait(1)
