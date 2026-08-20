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
        self.setup_layout("The Towers of Hanoi Rules", [
            "Towers of Hanoi has three rods.", 
            "Only one disk moves at once.", 
            "Larger disks cannot cover smaller ones."
        ])
        
        # Load assets
        rod_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")
        disk_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")

        # === Animation for Lecture Line 1 ===
        # Display three rods in #00FF00
        rods = VGroup(*[rod_img.copy().set_color("#00FF00") for _ in range(3)]).arrange(RIGHT, buff=1.0)
        self.place_at_grid(rods, 'C5', scale_factor=0.6)
        
        # Add labels for rods
        labels = VGroup(*[Text(l, font_size=18, color=WHITE) for l in ["Source", "Aux", "Target"]])
        for i, label in enumerate(labels):
            label.next_to(rods[i], DOWN, buff=0.1)
            self.add(label)
        
        self.add(rods)
        self.lecture[0].set_color("#00FF00")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show animation of moving one disk between rods, highlighting in #FF00FF
        disk = disk_img.copy().set_color("#FF00FF")
        self.place_at_grid(disk, 'C5', scale_factor=0.3)
        self.play(FadeIn(disk))
        self.play(disk.animate.move_to(self.grid['C3']))
        
        self.lecture[1].set_color("#FFFF00")
        move_text = Text("1 Move", color="#FFFF00", font_size=20)
        self.place_at_grid(move_text, 'D5', scale_factor=0.7)
        self.play(FadeIn(move_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Show error animation (disk collision) in #FF0000
        self.lecture[2].set_color("#FF0000")
        rule_text = Text("Large > Small", color="#FF0000", font_size=20)
        self.place_in_area(rule_text, 'D6', 'E6', scale_factor=0.6)
        self.play(FadeIn(rule_text))
        
        # Visual error flash
        error_flash = Rectangle(width=0.5, height=0.5, color="#FF0000", fill_opacity=0.5)
        error_flash.move_to(self.grid['C3'])
        self.play(Flash(error_flash, color="#FF0000"))
        self.wait(2)
