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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "Local Identity Generation: The Rotating IDs"
        lines = [
            "Each phone generates a secret daily master key.",
            "Hash functions derive temporary IDs from the master key.",
            "These ephemeral IDs change every fifteen minutes.",
            "Rotating IDs prevent tracking a user over time.",
            "All identity generation happens strictly on your device."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_SK = "#F39C12"
        COLOR_RPI1 = "#1ABC9C"
        COLOR_RPI2 = "#16A085"
        COLOR_RPI3 = "#138D75"
        COLOR_GREY = "#BDC3C7"

        # Assets
        KEY_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/key.svg"
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        # A golden 'Secret Key' (SK) (#F39C12) [Asset: ...] appears.
        self.lecture[0].set_color(COLOR_SK)
        
        sk_svg = SVGMobject(KEY_ASSET).set_color(COLOR_SK).scale(0.3)
        sk_text = Text("Secret Key (SK)", font_size=18, color=COLOR_SK)
        sk_group = VGroup(sk_svg, sk_text).arrange(DOWN, buff=0.2)
        # B3 position
        self.place_at_grid(sk_group, "B3", scale_factor=0.8)
        
        self.play(FadeIn(sk_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Arrow from SK to three 'RPI' boxes.
        # RPI_1 (#1ABC9C), RPI_2 (#16A085), RPI_3 (#138D75) appear.
        self.lecture[1].set_color(COLOR_RPI1)
        
        # Issue 39: Move RPIs to row E to avoid cramping
        rpi1_box = Rectangle(width=1.2, height=0.6, color=COLOR_RPI1, fill_opacity=0.2)
        rpi1_text = Text("RPI_1", font_size=16, color=COLOR_RPI1)
        rpi1 = VGroup(rpi1_box, rpi1_text)
        self.place_at_grid(rpi1, "E2", scale_factor=0.8)

        rpi2_box = Rectangle(width=1.2, height=0.6, color=COLOR_RPI2, fill_opacity=0.2)
        rpi2_text = Text("RPI_2", font_size=16, color=COLOR_RPI2)
        rpi2 = VGroup(rpi2_box, rpi2_text)
        self.place_at_grid(rpi2, "E3", scale_factor=0.8)

        rpi3_box = Rectangle(width=1.2, height=0.6, color=COLOR_RPI3, fill_opacity=0.2)
        rpi3_text = Text("RPI_3", font_size=16, color=COLOR_RPI3)
        rpi3 = VGroup(rpi3_box, rpi3_text)
        self.place_at_grid(rpi3, "E4", scale_factor=0.8)

        # Arrows from SK to RPIs
        arrow1 = Arrow(sk_group.get_bottom(), rpi1.get_top(), color=WHITE, buff=0.2)
        arrow2 = Arrow(sk_group.get_bottom(), rpi2.get_top(), color=WHITE, buff=0.2)
        arrow3 = Arrow(sk_group.get_bottom(), rpi3.get_top(), color=WHITE, buff=0.2)
        
        # Issue 38: Move hash_label to C4 to avoid overlapping central arrow
        hash_label = Text("Hash Function", font_size=14, color=WHITE)
        self.place_at_grid(hash_label, "C4", scale_factor=1.0)

        self.play(Create(arrow1), Create(arrow2), Create(arrow3), Write(hash_label))
        self.play(FadeIn(rpi1), FadeIn(rpi2), FadeIn(rpi3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Clock icon (#BDC3C7) spins; RPI_1 fades, RPI_2 glows.
        self.lecture[2].set_color(COLOR_GREY)
        
        # Issue 40: Move clock to E5 to align with RPI boxes
        clock_circle = Circle(radius=0.3, color=COLOR_GREY)
        clock_hand1 = Line(clock_circle.get_center(), clock_circle.get_top() * 0.8 + clock_circle.get_center() * 0.2, color=COLOR_GREY)
        clock_hand2 = Line(clock_circle.get_center(), clock_circle.get_right() * 0.6 + clock_circle.get_center() * 0.4, color=COLOR_GREY)
        clock = VGroup(clock_circle, clock_hand1, clock_hand2)
        self.place_at_grid(clock, "E5", scale_factor=0.8)

        self.play(FadeIn(clock))
        self.play(
            Rotate(clock_hand1, angle=-2*PI, about_point=clock_circle.get_center()),
            Rotate(clock_hand2, angle=-4*PI, about_point=clock_circle.get_center()),
            rpi1.animate.set_opacity(0.3),
            rpi2.animate.set_stroke(width=6).set_fill(opacity=0.6),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Rotating IDs prevent tracking a user over time.
        self.lecture[3].set_color(COLOR_RPI2)
        
        self.play(
            rpi2.animate.set_opacity(0.3).set_stroke(width=2).set_fill(opacity=0.2),
            rpi3.animate.set_stroke(width=6).set_fill(opacity=0.6),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A border (#BDC3C7) surrounds the key and IDs, labeled 'Local Device' [Asset: ...].
        self.lecture[4].set_color(COLOR_GREY)
        
        # Issue 32: Use phone asset and surrounding border
        device_border = DashedVMobject(RoundedRectangle(
            width=5.5, 
            height=5.0, 
            corner_radius=0.4, 
            color=COLOR_GREY
        ), num_dashes=50)
        
        self.place_in_area(device_border, "A1", "F6", scale_factor=1.0)
        
        phone_svg = SVGMobject(PHONE_ASSET).set_color(COLOR_GREY).scale(0.3)
        device_text = Text("Local Device", font_size=20, color=COLOR_GREY)
        label_group = VGroup(phone_svg, device_text).arrange(RIGHT, buff=0.2)
        # Position label at bottom of the "device" area
        self.place_at_grid(label_group, "F3", scale_factor=1.0)
        
        self.play(Create(device_border))
        self.play(FadeIn(label_group))
        self.wait(2)
