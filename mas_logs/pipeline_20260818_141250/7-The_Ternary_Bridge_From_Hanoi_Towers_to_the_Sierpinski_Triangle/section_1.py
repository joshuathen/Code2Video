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
            "Ternary systems utilize three states: zero, one, two.",
            "Three states naturally represent transitions between three locations.",
            "Base-three counting simplifies Hanoi Tower movement patterns."
        ]
        self.setup_layout("The Base-3 Intuition", lecture_lines)

        colors = ["#FF5733", "#33FF57", "#3357FF"]
        # Use SVG asset as requested
        disks = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg", color=c) for c in colors])
        labels = VGroup(*[Text(f"Size {i}", font_size=16) for i in range(3)])

        # Group them for easier manipulation if needed
        group_disks_labels = VGroup()
        for i in range(3):
            group_disks_labels.add(VGroup(disks[i], labels[i]))

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(disks))
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        # Positioning based on VideoCritic requests
        self.place_at_grid(disks[0], 'B2', scale_factor=0.8)
        self.place_at_grid(disks[1], 'C2', scale_factor=0.8)
        self.place_at_grid(disks[2], 'D2', scale_factor=0.8)
        
        self.place_at_grid(labels[0], 'B3', scale_factor=0.5)
        self.place_at_grid(labels[1], 'C3', scale_factor=0.5)
        self.place_at_grid(labels[2], 'D3', scale_factor=0.5)
        
        self.play(*[Write(l) for l in labels])

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Shift disks to simulate transition
        self.play(
            disks[0].animate.move_to(self.grid["B5"]),
            disks[1].animate.move_to(self.grid["C5"]),
            disks[2].animate.move_to(self.grid["D5"]),
            run_time=1.5
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF69B4"))
        self.play(
            *[disk.animate.set_color("#FFFFFF") for disk in disks],
            run_time=1
        )
        self.wait(1)
